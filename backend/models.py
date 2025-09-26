from pydantic import BaseModel

class RequestData(BaseModel):
    question: str

class RequestReg(BaseModel):
    login: str
    password: str
    role: str

class RequestAuth(BaseModel):
    login: str
    password: str