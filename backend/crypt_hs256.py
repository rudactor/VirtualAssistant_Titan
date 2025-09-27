from passlib.context import CryptContext
import hashlib
import bcrypt
import jwt
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import os

load_dotenv()

class Crypt(object):
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.SECRET_KEY = os.environ.get("SECRET_KEY")
        self.ALGORITHM = os.environ.get("ALGORITHM")
    
    def decrypt(self, password: str, hashed: str):
        sha = hashlib.sha256(password.encode("utf-8")).digest()
        return bcrypt.checkpw(sha, hashed.encode())
    
    def encrypt(self, password: str):
        sha = hashlib.sha256(password.encode("utf-8")).digest()
        hashed = bcrypt.hashpw(sha, bcrypt.gensalt())
        return hashed.decode()
    
    def _check_algorithm(self):
        if self.ALGORITHM not in jwt.algorithms.get_default_algorithms():
            raise ValueError(f"Algorithm {self.ALGORITHM} is not supported by PyJWT")
    
    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt
    
    def decode_access_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")