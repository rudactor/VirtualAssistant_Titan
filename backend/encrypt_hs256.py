from passlib.context import CryptContext

class Encrypt(object):
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def decrypt(self, password: str):
        return self.pwd_context.hash(password)