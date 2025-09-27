import sqlite3

class SqliteDatabase(object):
    
    def __init__(self):
        self.conn = sqlite3.connect("sqlite.sqlite3")
        self.cursor = self.conn.cursor()
        self._create_database()

    def _create_database(self):
        self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    login TEXT,
                    hash_password TEXT
                )
            ''')

        self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id_user INTEGER PRIMARY KEY,
                    message TEXT,
                    is_answer BOOLEAN
                )
            ''')
        
    def _add_data(self, table: str, data: dict):
        keys = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        values = tuple(data.values())
        self.cursor.execute(f"INSERT INTO {table} ({keys}) VALUES ({placeholders})", values)
        self.conn.commit()
        
    def _get_data(self, table: str) -> list:
        data = self.cursor.execute(f'''
        SELECT * FROM {table}
                            '''
                            )
        return data.fetchall()
    
    def close(self):
        self.conn.close()
        
data = {
    "login": '12345',
    "hash_password": '123513',
    "role": '135513'
}
        
# # print(db._add_data("users", data))
# print(db._get_data("users"))

# db.close()