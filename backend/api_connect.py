from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from starlette import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from models_types import RequestAuth, RequestData, RequestReg
from crypt_hs256 import Crypt
from dotenv import load_dotenv
from sqlite_connect import SqliteDatabase
# from main import main
import os

load_dotenv()

class BackendApp(object):
    def __init__(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3002"],  # фронтенд
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.db = SqliteDatabase()
        self.crypt = Crypt()
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")
        self.init_routes()
        
        self.request = ""

    def init_routes(self):
        # self.app.get('/') (self.root)                           # получение ответа от ассистента
        self.app.post("/reg") (self.registration)               # регистрация   
        self.app.post("/auth") (self.authorization)             # авторизация
        self.app.post('/check') (self.send_request)
         # self.app.post("/send") (self.create_request)            # отправить вопрос ассистенту
    
    # async def root(self):
    #     self.answer = main(self.request) if self.request else 'no yet'
    #     return JSONResponse({"message": self.answer}, status_code=status.HTTP_200_OK)

    async def registration(self, data: RequestReg):
        if len(data.login) <= 4 or len(data.password) <= 8:
            return JSONResponse(content={"message": "small length login(>4 length) or password(>8 length)"}, status_code=status.HTTP_400_BAD_REQUEST)
        
        hash_password = self.crypt.encrypt(data.password)
        data = data.dict()
        data = {"login": data["login"], "hash_password": hash_password}
        self.db._add_data("users", data)
        token = self.crypt.create_access_token({"sub": data['login']})
        return JSONResponse(content={"message": "successfully", "data": data, 'token': token}, status_code=status.HTTP_200_OK)
    
    async def authorization(self, data: RequestAuth):
        all_data = self.db._get_data("users")
        user = next((u for u in all_data if u[1] == data.login), None)

        if user and self.crypt.decrypt(data.password, user[2]):
            token = self.crypt.create_access_token({"id": user[0], "sub": user[1]})
            return JSONResponse(content={"message": "you are authorized", "token": token}, status_code=status.HTTP_200_OK)
        return JSONResponse(content={"message": "login or password are not a correct"}, status_code=status.HTTP_200_OK)
    
    async def get_current_user(self, token: str):
        if token == '':
            token = Depends(self.oauth2_scheme)
        try:
            payload = self.crypt.decode_access_token(token)
            return payload
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
    async def send_request(self, data: RequestData, user):
        user = Depends(self.get_current_user(self))
        return {"message": f"Request received from {user['sub']}"}
    
    # async def create_request(self, data: RequestData):
    #     self.request = data.question
    #     return JSONResponse(content={"message": "request saved"}, status_code=status.HTTP_200_OK)
        
backend = BackendApp()
app = backend.app