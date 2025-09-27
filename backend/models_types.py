from pydantic import BaseModel

class RequestData(BaseModel):
    question: str
    chat_id: int

class RequestReg(BaseModel):
    login: str
    password: str

class RequestChat(BaseModel):
    title: str | None

class RequestAuth(BaseModel):
    login: str
    password: str
    
class RequestAddChat(BaseModel):
    chat_id: int
    user_id: int
    
class RequestMessage(BaseModel):
    chat_id: int