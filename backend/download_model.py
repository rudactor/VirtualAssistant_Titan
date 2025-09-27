# download_model.py
import os
import time
import math
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry

OUT_DIR   = "./models"
FILENAME  = "qwen2.5-3b-instruct-q4_k_m.gguf"

URLS = [
    # 1) зеркало
    f"https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/{FILENAME}",
    # 2) основной HF
    f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/{FILENAME}",
]

CHUNK_SIZE = 1024 * 1024  # 1 MB
TIMEOUT = 30              # сек на запрос
MAX_RETRIES_PER_URL = 5   # попыток на один URL
BACKOFF_BASE = 2          # экспоненциальная задержка 1,2,4,8,...

os.makedirs(OUT_DIR, exist_ok=True)
DST_PATH = os.path.join(OUT_DIR, FILENAME)

def make_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s

def head_content_length(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    # fallback — попробуем GET без тела (не всегда работает, но иногда даёт размер)
    try:
        r = session.get(url, stream=True, timeout=TIMEOUT)
        if r.status_code in (200, 206) and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    return None

def download_with_resume(url: str, dst: str) -> bool:
    session = make_session()
    total = head_content_length(session, url)
    # если сервер не отдал размер — всё равно попытаемся
    current = os.path.getsize(dst) if os.path.exists(dst) else 0
    desc = os.path.basename(dst)

    # если полный файл уже есть
    if total is not None and current >= total > 0:
        print(f"[download_model] already complete: {dst} ({total} bytes)")
        return True

    # прогресс-бар с учётом докачки
    bar_total = total if total is not None else 0
    bar = tqdm(
        total=bar_total,
        initial=current if total is not None else 0,
        unit="B", unit_scale=True, unit_divisor=1024,
        desc=desc, ascii=True
    )

    tries = 0
    ok = False
    try:
        mode = "ab" if current > 0 else "wb"
        with open(dst, mode) as f:
            while True:
                headers = {}
                if current > 0:
                    headers["Range"] = f"bytes={current}-"
                try:
                    with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                        if r.status_code in (200, 206):
                            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                current += len(chunk)
                                if total is not None:
                                    bar.update(len(chunk))
                                # если знаем общий размер и догрузили всё — выходим
                                if total is not None and current >= total:
                                    ok = True
                                    break
                            # если сервер закрыл соединение до конца — повторим, сохранив current
                            if total is None:
                                # нет известного размера: если поток закончился — считаем успех
                                ok = True
                                break
                            if current >= total:
                                ok = True
                                break
                        else:
                            tries += 1
                            if tries > MAX_RETRIES_PER_URL:
                                break
                            sleep_s = BACKOFF_BASE ** (tries - 1)
                            time.sleep(sleep_s)
                            continue
                except (requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.ReadTimeout) as e:
                    tries += 1
                    if tries > MAX_RETRIES_PER_URL:
                        print(f"[download_model] giving up on this URL after {tries} attempts: {e}")
                        break
                    sleep_s = BACKOFF_BASE ** (tries - 1)
                    time.sleep(sleep_s)
                    # цикл продолжится, откроем новое соединение с Range с текущего места
                    continue
    finally:
        bar.close()

    # двойная проверка завершения по размеру, если он известен
    if ok and total is not None:
        final_size = os.path.getsize(dst)
        if final_size < total:
            print(f"[download_model] warning: expected {total}, got {final_size} — will try next URL")
            return False
    return ok

def main():
    if os.path.exists(DST_PATH):
        print(f"[download_model] existing file: {DST_PATH} ({os.path.getsize(DST_PATH)} bytes)")

    for idx, url in enumerate(URLS, start=1):
        print(f"[download_model] ({idx}/{len(URLS)}) downloading {url} -> {DST_PATH}")
        ok = download_with_resume(url, DST_PATH)
        if ok:
            print(f"[download_model] done: {DST_PATH}")
            return
        else:
            print(f"[download_model] failed on: {url} — trying next mirror...")

    print("[download_model] ERROR: all mirrors failed. "
          "Скачай вручную (VPN/зеркало) и положи файл в ./models/")

if __name__ == "__main__":
    main()
