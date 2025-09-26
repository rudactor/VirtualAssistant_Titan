from passlib.context import CryptContext

class Decrypt(object):
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def decrypt(self, plain_password: str, hashed_password: str):
        return self.pwd_context.verify(plain_password, hashed_password)