import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

import psutil


ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
PROJECT = Path(r"D:\DA\DoAn_TTNT_De19\adversarial-rl-testgen")
TOOLS = ASSETS / "tools/gui"
STATE = TOOLS / "services.json"
URL = "http://127.0.0.1:7860/?__theme=light"


def read_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"processes": []}


def owned_process(record):
    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["create_time"]) > 0.1:
            return None
        if process.cmdline() != record["command"]:
            return None
        return process
    except (psutil.Error, KeyError):
        return None


def healthy():
    state = read_state()
    return len(state["processes"]) == 3 and all(owned_process(record) is not None for record in state["processes"])


def stop():
    for record in reversed(read_state()["processes"]):
        process = owned_process(record)
        if process is None:
            continue
        children = process.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            process.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs(children + [process], timeout=5)
        for item in alive:
            try:
                item.kill()
            except psutil.Error:
                pass
    STATE.write_text(json.dumps({"processes": [], "stopped_at": time.time()}), encoding="utf-8")
    print("Da dong giao dien va cac tien trinh do bo cong cu nay khoi chay.")


def ready(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def start(open_browser):
    if healthy():
        print("Giao dien dang chay:", URL)
        if open_browser:
            webbrowser.open(URL)
        return
    stop()
    for port in (7860, 6006):
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("Cong " + str(port) + " dang duoc ung dung khac su dung. Khong tu dong dong ung dung do.")
    TOOLS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "temp/gradio").mkdir(parents=True, exist_ok=True)
    (ASSETS / "evidence/tensorboard").mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", GRADIO_ANALYTICS_ENABLED="False",
                       TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"), GRADIO_TEMP_DIR=str(ASSETS / "temp/gradio"), MPLCONFIGDIR=str(ASSETS / "cache/matplotlib-ui"))
    commands = [
        [sys.executable, str(PROJECT / "scripts/export_tensorboard.py"), "--watch"],
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(ASSETS / "evidence/tensorboard"), "--host", "127.0.0.1", "--port", "6006", "--load_fast=false", "--reload_interval", "5"],
        [sys.executable, str(PROJECT / "scripts/ui_dashboard.py")],
    ]
    state = {"processes": [], "started_at": time.time(), "url": URL, "tensorboard_url": "http://127.0.0.1:6006"}
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        for name, command in zip(("exporter", "tensorboard", "dashboard"), commands, strict=True):
            with (TOOLS / (name + ".log")).open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(PROJECT), env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            owned = psutil.Process(process.pid)
            state["processes"].append({"name": name, "pid": process.pid, "create_time": owned.create_time(), "command": command})
            STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        for _ in range(60):
            if ready("http://127.0.0.1:7860/config") and ready("http://127.0.0.1:6006"):
                print("Gradio:", URL)
                print("TensorBoard: http://127.0.0.1:6006")
                if open_browser:
                    webbrowser.open(URL)
                return
            if not healthy():
                raise RuntimeError("Mot dich vu da dung som. Xem log tai " + str(TOOLS))
            time.sleep(1)
        raise RuntimeError("Dich vu chua san sang. Xem log tai " + str(TOOLS))
    except BaseException:
        stop()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    if args.action == "start":
        start(not args.no_browser)
        if args.hold:
            try:
                while healthy():
                    time.sleep(1)
            except KeyboardInterrupt:
                stop()
    elif args.action == "stop":
        stop()
    else:
        print(json.dumps({"healthy": healthy(), "state": read_state()}, indent=2))


if __name__ == "__main__":
    main()
