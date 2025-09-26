from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette import status
from pydantic import BaseModel
from decrypt_hs256 import Decrypt
from encrypt_hs256 import Encrypt
from dotenv import load_dotenv
from connect_to_db import DataBase
from main import main
import os
from datetime import datetime, timedelta, timezone

load_dotenv()

class BackendApp(object):
    def __init__(self):
        self.app = FastAPI()
        self.db = DataBase()
        self.decrypt = Decrypt()
        self.encrypt = Encrypt()
        self.init_routes()
        self.SECRET_KEY = os.environ.get("SECRET_KEY")
        self.ALGORITHM = os.environ.get("ALGORITHM")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
        
        self.request = ""

    def init_routes(self):
        self.app.get('/') (self.root)
        self.app.post("/registr") (self.registration)
        self.app.post("/auth") (self.authorization)
        self.app.post("/send") (self.create_request)
    
    async def root(self):
        self.answer = main(self.request) if self.request else 'no yet'
        return JSONResponse({"message": self.answer}, status_code=status.HTTP_200_OK)

    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    async def registration(self, data: RequestReg):
        self.db._add_data(data.dict())
        return JSONResponse(content={"message": "successfully"}, status_code=status.HTTP_200_OK)
    
    async def authorization(self, data: RequestAuth):
        all_data = self.db._get_data("*")
        user = next((u for u in all_data if u["login"] == data.login and u["password"] == data.password), None)

        if user:
            token = self.create_access_token({"sub": user["login"], "role": user["role"]})
            print(token)
            return JSONResponse(content={"message": "you are authorized", "token": token}, status_code=status.HTTP_200_OK)
        else:
            return JSONResponse(content={"message": "login or password are not a correct"}, status_code=status.HTTP_200_OK)
    
    async def create_request(self, data: RequestData):
        self.request = data.question
        return JSONResponse(content={"message": "request saved"}, status_code=status.HTTP_200_OK)
        
backend = BackendApp()
app = backend.app