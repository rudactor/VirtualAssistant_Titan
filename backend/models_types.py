from pydantic import BaseModel

class RequestData(BaseModel):
    question: str
    chat_id: int

class RequestReg(BaseModel):
    login: str
    password: str

class RequestAuth(BaseModel):
    login: str
    password: str
    
class RequestAddChat(BaseModel):
    title: str | None
    user_id: int
    
class RequestMessage(BaseModel):
    chat_id: int
    
class RequestAllChats(BaseModel):
    user_id: int