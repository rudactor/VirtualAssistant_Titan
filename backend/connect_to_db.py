from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

class DataBase(object):
    def __init__(self) -> None:
        try:
            self.URL: str = os.environ.get("SUPABASE_URL")
            self.KEY: str = os.environ.get("SUPABASE_KEY")
            self.client: Client = create_client(self.URL, self.KEY)
        except Exception as e:
            print(e)
        
    def _get_data(self, data: str) -> str:
        try:
            response = (
                self.client.table('workers')
                .select(data)
                .execute()
            )
            return response.data
        except Exception as e:
            print(e)
            
    def _add_data(self, data: str) -> str:
        try:
            response = (
                self.client.table('workers')
                .insert(data)
                .execute()
            )
            return response.data
        except Exception as e:
            print(e)
            
# db = DataBase()
# db._add_data({
#     "login": '12345',
#     "password": "12353151",
#     "role": "worker"
# })