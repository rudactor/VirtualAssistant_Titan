from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from starlette import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from models_types import *
from crypt_hs256 import Crypt
from dotenv import load_dotenv
from sqlite_connect import SqliteDatabase
from chat_engine import ask, start_chat, get_messages, show_chats
import os

load_dotenv()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = Crypt().decode_access_token(token)
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

class BackendApp(object):
    def __init__(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],  # фронтенд
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.db = SqliteDatabase()
        self.crypt = Crypt()
        self.init_routes()
        
        self.request = ""

    def init_routes(self):
        self.app.post('/') (self.root)                           # получение ответа от ассистента
        self.app.post("/reg") (self.registration)               # регистрация   
        self.app.post("/auth") (self.authorization)             # авторизация
        self.app.post('/check') (self.send_request)           # проверка доступа
        self.app.post("/add_chat") (self._add_chat_to_user)
        self.app.post('/messages') (self._get_messages)
        self.app.post('/get_chats') (self._get_chats)
    
    async def root(self, data: RequestData):
        self.answer = ask(data.chat_id, data.question)
        return JSONResponse({"message": self.answer}, status_code=status.HTTP_200_OK)

    async def registration(self, data: RequestReg):
        if len(data.login) <= 4 or len(data.password) <= 8:
            return JSONResponse(content={"message": "small length login(>4 length) or password(>8 length)"}, status_code=status.HTTP_400_BAD_REQUEST)
        all_data = self.db._get_data("users")
        if next((u for u in all_data if u[1] == data.login), None):
            return JSONResponse(content={"message": "this user already have"}, status_code=status.HTTP_400_BAD_REQUEST)
        
        hash_password = self.crypt.encrypt(data.password)
        data = data.dict()
        data = {"login": data["login"], "hash_password": hash_password}
        self.db._add_data("users", data)
        token = self.crypt.create_access_token({'id': len(all_data) + 1, "sub": data['login']})
        return JSONResponse(content={"message": "successfully", "data": data, 'token': token}, status_code=status.HTTP_200_OK)
    
    async def authorization(self, data: RequestAuth):
        all_data = self.db._get_data("users")
        user = next((u for u in all_data if u[1] == data.login), None)

        if user and self.crypt.decrypt(data.password, user[2]):
            token = self.crypt.create_access_token({"id": user[0], "sub": user[1]})
            return JSONResponse(content={"message": "you are authorized", "token": token}, status_code=status.HTTP_200_OK)
        return JSONResponse(content={"message": "login or password are not a correct"}, status_code=status.HTTP_200_OK)
        
    async def _add_chat_to_user(self, data: RequestAddChat):
        chat_id = start_chat(data.title)
        self.db._update_data(chat_id, data.user_id)
        return JSONResponse(content={"message": "added succesfully"}, status_code=status.HTTP_200_OK)
    
    async def _get_messages(self, data: RequestMessage):
        data = get_messages(data.chat_id)
        return JSONResponse(content={"data": data}, status_code=status.HTTP_200_OK)
    
    async def _get_chats(self, data: RequestAllChats):
        if self.db.get_chats(data.user_id)[0][0] == None:
            return JSONResponse(content={'data': []}, status_code=status.HTTP_200_OK)
        result = [int(i) for i in self.db.get_chats(data.user_id)[0][0].split(',') if i != '']
        data = [show_chats(id_chat) for id_chat in result]
        return JSONResponse(content={'data': data}, status_code=status.HTTP_200_OK)
        
    async def send_request(self, user=Depends(get_current_user)):
        return {"message": user['id']}

        
backend = BackendApp()
app = backend.app