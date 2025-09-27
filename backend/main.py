# main.py
import os, sys, time

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from chat_engine import (
    init_db, start_chat, ask, list_chats, chat_exists, debug_find
)
from ingest import ingest

def pick(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""

def _choose_chat() -> int:
    rows = list_chats(50)
    if not rows:
        print("Список пуст. Создаю новый.")
        return start_chat("Диалог")
    print("\nДоступные чаты:")
    for cid, title, ts in rows:
        print(f"  {cid}: {title} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))})")
    val = pick("Введите chat_id (или Enter для нового): ").strip()
    if val.isdigit():
        cid = int(val)
        if chat_exists(cid):
            return cid
    return start_chat("Диалог")

def main():
    init_db()

    print("RAG загрузить документы сейчас? [y/N] ", end="")
    if (pick("") or "n").lower().startswith("y"):
        ingest(scope="global")

    print("\nНовый чат? [Y/n] ", end="")
    if not (pick("") or "y").lower().startswith("n"):
        title = pick("Название чата (enter — по умолчанию): ") or "Новый диалог"
        chat_id = start_chat(title)
    else:
        chat_id = _choose_chat()

    print(f"\nchat_id={chat_id}. Введите вопрос (пустая строка — выход).")
    print("Команды: ':find ваш_запрос' — показать кандидаты; ':chats' — список; ':open id' — открыть чат.")
    while True:
        q = pick("> ").strip()
        if not q:
            break

        if q.startswith(":find "):
            query = q[len(":find "):].strip()
            print(debug_find(query, scope=str(chat_id)))
            print()
            continue

        if q == ":chats":
            for cid, title, ts in list_chats(50):
                print(f"  {cid}: {title} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))})")
            print()
            continue

        if q.startswith(":open "):
            val = q.split(" ", 1)[1].strip()
            if val.isdigit() and chat_exists(int(val)):
                chat_id = int(val)
                print(f"Открыт чат {chat_id}")
            else:
                print("Несуществующий chat_id")
            continue

        a = ask(chat_id, q)
        print("\n— Ответ —")
        print(a)
        print()

if __name__ == "__main__":
    main()