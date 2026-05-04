from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_REPO_ID = "tencent/HY-MT1.5-7B-GGUF"
DEFAULT_FILENAME = "HY-MT1.5-7B-Q4_K_M.gguf"
DEFAULT_LOCAL_FILE = Path("runtimes/models/llm/hy-mt1.5-7b.gguf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--local-file", default=str(DEFAULT_LOCAL_FILE))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_id = str(args.repo_id).strip()
    filename = str(args.filename).strip()
    target = Path(args.local_file).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"[runtime:llm_model] download {repo_id}/{filename}")
    try:
        req = Request(url=url, headers={"User-Agent": "SyncTranslate-runtime-setup"})
        with urlopen(req, timeout=60) as response, tmp.open("wb") as fp:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            next_report = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fp.write(chunk)
                downloaded += len(chunk)
                if total and downloaded >= next_report:
                    percent = downloaded * 100 / total
                    print(f"[runtime:llm_model] {percent:5.1f}%")
                    next_report = downloaded + max(total // 20, 64 * 1024 * 1024)
    except (HTTPError, URLError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(f"Failed to download LLM model from {url}: {exc}") from exc

    tmp.replace(target)
    print(f"LLM model ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
