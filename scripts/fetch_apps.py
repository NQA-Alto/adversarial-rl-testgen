import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone

from apps_core import REVISION, SOURCE, digest, write


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    SOURCE.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "codeparrot/apps", "revision": REVISION,
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "files": [], "official_test_downloaded": False}
    for name in ("README.md", "apps.py", "train.jsonl"):
        path = SOURCE / name
        url = f"https://huggingface.co/datasets/codeparrot/apps/resolve/{REVISION}/{name}"
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            partial.rename(path)
        manifest["files"].append({"name": name, "url": url, "bytes": path.stat().st_size, "sha256": digest(path)})
    target = SOURCE / "source-manifest.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing["files"] != manifest["files"]:
            raise RuntimeError("Source differs from its saved manifest")
    else:
        write(target, manifest)
    print("APPS source verified:", SOURCE)


if __name__ == "__main__":
    main()
