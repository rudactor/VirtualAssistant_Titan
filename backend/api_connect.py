from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette import status
from models_types import RequestAuth, RequestData, RequestReg
from crypt_hs256 import Crypt
from dotenv import load_dotenv
from connect_to_db import DataBase
from main import main
import os

load_dotenv()

class BackendApp(object):
    def __init__(self):
        self.app = FastAPI()
        self.db = DataBase()
        self.crypt = Crypt()
        self.init_routes()
        
        self.request = ""

    def init_routes(self):
        self.app.get('/') (self.root)                           # получение ответа от ассистента
        self.app.post("/reg") (self.registration)               # регистрация   
        self.app.post("/auth") (self.authorization)             # авторизация
        self.app.post("/send") (self.create_request)            # отправить вопрос ассистенту
    
    async def root(self):
        self.answer = main(self.request) if self.request else 'no yet'
        return JSONResponse({"message": self.answer}, status_code=status.HTTP_200_OK)

    async def registration(self, data: RequestReg):
        if len(data.login) <= 4 or len(data.password) <= 8:
            return JSONResponse(content={"message": "small length login(>4 length) or password(>8 length)"}, status_code=status.HTTP_400_BAD_REQUEST)
        
        hash_password = self.crypt.encrypt(data.password)
        data = data.dict()
        data = {"login": data["login"], "role": data["role"], "hashed_password": hash_password}
        self.db._add_data(data)
        return JSONResponse(content={"message": "successfully", "data": data}, status_code=status.HTTP_200_OK)
    
    async def authorization(self, data: RequestAuth):
        all_data = self.db._get_data("*")
        user = next((u for u in all_data if u["login"] == data.login), None)

        if user and self.crypt.decrypt(data.password, user["hashed_password"]):
            token = self.crypt.create_access_token({"sub": user["login"], "role": user["role"]})
            return JSONResponse(content={"message": "you are authorized", "token": token}, status_code=status.HTTP_200_OK)
        else:
            return JSONResponse(content={"message": "login or password are not a correct"}, status_code=status.HTTP_200_OK)
    
    async def create_request(self, data: RequestData):
        self.request = data.question
        return JSONResponse(content={"message": "request saved"}, status_code=status.HTTP_200_OK)
        
backend = BackendApp()
app = backend.app