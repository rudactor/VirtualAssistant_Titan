import os
import time
import requests
from pathlib import Path
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry

BACKEND_DIR = Path(__file__).resolve().parent
OUT_DIR = BACKEND_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
DST_PATH = OUT_DIR / FILENAME

# основная ссылка + зеркало
URLS = [
    f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/{FILENAME}",
    f"https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/{FILENAME}",
]

CHUNK = 1024 * 1024
TIMEOUT = 30
MAX_RETRIES_PER_URL = 5
BACKOFF_BASE = 2

def session():
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"], raise_on_status=False,
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s

def head_len(s: requests.Session, url: str):
    try:
        r = s.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.ok and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    return None

def download_resume(url: str, dst: Path) -> bool:
    s = session()
    total = head_len(s, url)
    have = dst.stat().st_size if dst.exists() else 0

    # если уже скачано
    if total and have >= total:
        print(f"[download_model] already complete: {dst} ({total} bytes)")
        return True

    desc = dst.name
    with tqdm(total=total or 0, initial=have if total else 0,
              unit="B", unit_scale=True, unit_divisor=1024,
              desc=desc, ascii=True) as bar:
        tries = 0
        mode = "ab" if have else "wb"
        with open(dst, mode) as f:
            while True:
                headers = {"Range": f"bytes={have}-"} if have else {}
                try:
                    with s.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                        if r.status_code not in (200, 206):
                            tries += 1
                            if tries > MAX_RETRIES_PER_URL: return False
                            time.sleep(BACKOFF_BASE ** (tries - 1)); continue

                        for chunk in r.iter_content(chunk_size=CHUNK):
                            if not chunk: continue
                            f.write(chunk); have += len(chunk)
                            if total: bar.update(len(chunk))
                            if total and have >= total: return True

                        # если сервер оборвал без total — считаем успех
                        if not total: return True

                except (requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.ReadTimeout):
                    tries += 1
                    if tries > MAX_RETRIES_PER_URL: return False
                    time.sleep(BACKOFF_BASE ** (tries - 1))
                    continue

def main():
    print(f"[download_model] target: {DST_PATH}")
    if DST_PATH.exists():
        print(f"[download_model] existing file size: {DST_PATH.stat().st_size} bytes")
    for i, url in enumerate(URLS, 1):
        print(f"[download_model] ({i}/{len(URLS)}) {url}")
        if download_resume(url, DST_PATH):
            print(f"[download_model] done: {DST_PATH}")
            return
        print("[download_model] switching mirror…")
    print("[download_model] ERROR: all mirrors failed. "
          f"Download manually and place file at: {DST_PATH}")

if __name__ == "__main__":
    main()
