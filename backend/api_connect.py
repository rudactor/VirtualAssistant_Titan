from fastapi import FastAPI
from starlette import status
from starlette.responses import Response
from pydantic import BaseModel
from connect_to_db import DataBase
from main import main

class RequestData(BaseModel):
    question: str

class RequestAuth(BaseModel):
    login: str
    password: str
    role: str

class BackendApp(object):
    def __init__(self):
        self.app = FastAPI()
        self.db = DataBase()
        self.init_routes()
        self.request = ""

    def init_routes(self):
        self.app.get('/') (self.root)
        self.app.get("/registr") (self.registration)
        self.app.post("/send") (self.create_request)
    
    async def root(self):
        self.answer = main(self.request)
        return {"message": self.answer}, Response(status_code=status.HTTP_200_OK)

    async def registration(self, data: RequestData):
        self.db._get_data(data)
        return {"message": "successfully"}, Response(status_code=status.HTTP_200_OK)

    async def create_request(self, data: RequestData):
        self.request = data.question
        return Response(status_code=status.HTTP_200_OK)
        
backend = BackendApp()
app = backend.app