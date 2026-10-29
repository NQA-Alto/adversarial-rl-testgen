import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from apps_core import ASSETS, digest, read, write


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = ASSETS / "env/Scripts/python.exe"
TASKS = ["Kiểm tra dữ liệu", "Chấm Qwen gốc trên Validation", "Huấn luyện GRPO trên Train", "Chấm adapter hiện tại trên Validation", "Huấn luyện SFT + GRPO trên APPS"]
CURRENT = None
GUARD = threading.Lock()


def command_for(task, steps):
    if task not in TASKS:
        raise ValueError("Tác vụ không hợp lệ.")
    if task == TASKS[0]:
        return [str(PYTHON), "-B", str(PROJECT / "scripts/verify_apps.py")], False
    if task in (TASKS[2], TASKS[4]) and (isinstance(steps, bool) or int(steps) != steps or not 1 <= int(steps) <= 96):
        raise ValueError("Số bước GRPO phải từ 1 đến 96.")
    if task == TASKS[4]:
        return [str(PYTHON), "-B", str(PROJECT / "scripts/apps_round.py"), "--grpo-steps", str(int(steps))], False
    command = [str(PYTHON), "-B", str(PROJECT / "scripts/apps_model_runner.py")]
    if task == TASKS[2]:
        command += ["train", "--steps", str(int(steps))]
        state = read(PROJECT / "APPS_STATE.json")
        parent = state.get("selected_sft_training") or state.get("sft_training")
        if parent:
            if digest(Path(parent["adapter_path"]) / "adapter_model.safetensors") != parent["adapter_sha256"]:
                raise ValueError("Adapter SFT đã thay đổi.")
            baseline = state.get("selected_sft_train_probe") or state["sft_validation"]
            command += ["--init-adapter", parent["adapter_path"], "--baseline-run", baseline["path"]]
    else:
        command += ["evaluate"]
        if task == TASKS[3]:
            record = read(PROJECT / "APPS_STATE.json").get("pilot_training")
            if not record:
                raise ValueError("Chưa có adapter APPS để chấm.")
            adapter = Path(record["adapter_path"])
            if digest(adapter / "adapter_model.safetensors") != record["adapter_sha256"]:
                raise ValueError("Adapter đã thay đổi so với bản lưu.")
            command += ["--adapter", str(adapter)]
    return command, True


def snapshot():
    with GUARD:
        job = CURRENT
        if job is None:
            records = sorted((ASSETS / "evidence/ui-jobs").glob("apps-*/job.json"), reverse=True)
            if not records:
                return "Chưa có tác vụ đang chạy.", "", ""
            record = records[0]
            status, folder = read(record)["status"], record.parent
        else:
            status, folder = job["status"], job["folder"]
    log = folder / "console.txt"
    text = log.read_text(encoding="utf-8", errors="replace")[-24000:] if log.exists() else ""
    return status, text, str(folder)


def worker(job, shared_lock, needs_audit):
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                       TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"), HF_HOME=str(ASSETS / "cache/huggingface"))
    command = job["command"]
    outcome = "failed"
    try:
        for stage in ("run", "audit") if needs_audit else ("run",):
            with GUARD:
                if job["stopping"]:
                    outcome = "cancelled"
                    break
                job["status"] = ("Đang chạy · " if stage == "run" else "Đang kiểm chứng kết quả · ") + job["task"]
                with (job["folder"] / "console.txt").open("a", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                job["process"] = process
            code = process.wait()
            if job["stopping"]:
                outcome = "cancelled"
                break
            if code != 0:
                raise RuntimeError("Tác vụ dừng với mã " + str(code))
            outcome = "completed"
            if stage == "run" and needs_audit:
                lines = (job["folder"] / "console.txt").read_text(encoding="utf-8").splitlines()
                locations = [Path(line.removeprefix("EVIDENCE: ").strip()) for line in lines if line.startswith("EVIDENCE: ")]
                if len(locations) != 1 or not locations[0].resolve().is_relative_to((ASSETS / "evidence").resolve()):
                    raise RuntimeError("Không xác định được thư mục kết quả để kiểm chứng.")
                job["evidence"] = str(locations[0])
                command = [str(PYTHON), "-B", str(PROJECT / "scripts/audit_apps_run.py"), str(locations[0])]
    except Exception as error:
        outcome = "failed"
        job["error"] = str(error)
    finally:
        try:
            with GUARD:
                job["status"] = {"completed": "Hoàn tất · ", "cancelled": "Đã dừng · ", "failed": "Chưa hoàn tất · "}[outcome] + job["task"]
                if job.get("error"):
                    job["status"] += " · " + job["error"]
                job["process"] = None
                job["done"] = True
                record = {key: value for key, value in job.items() if key != "process"}
                record["folder"] = str(record["folder"])
                record.update(outcome=outcome, ended_at=datetime.now().astimezone().isoformat())
                write(job["folder"] / "job.json", record)
        finally:
            shared_lock.release()


def start(task, steps, shared_lock, busy):
    global CURRENT
    if not shared_lock.acquire(blocking=False):
        status, text, folder = snapshot()
        return "Một lượt chạy đang hoạt động. " + status, text, folder
    try:
        if busy():
            raise ValueError("GPU đang có tác vụ APPS. Hãy đợi tác vụ đó hoàn tất.")
        command, needs_audit = command_for(task, steps)
        folder = ASSETS / "evidence/ui-jobs" / ("apps-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        folder.mkdir(parents=True, exist_ok=False)
        job = {"task": task, "command": command, "folder": folder, "status": "Đang chuẩn bị · " + task,
               "started_at": datetime.now().astimezone().isoformat(), "stopping": False, "process": None, "done": False}
        write(folder / "plan.json", {"task": task, "command": command, "started_at": job["started_at"]})
        with GUARD:
            CURRENT = job
        threading.Thread(target=worker, args=(job, shared_lock, needs_audit), daemon=True).start()
    except Exception as error:
        shared_lock.release()
        return str(error), "", ""
    return snapshot()


def stop():
    import psutil
    with GUARD:
        job = CURRENT
        if job is None or job["done"]:
            return snapshot_without_lock(job)
        job["stopping"] = True
        job["status"] = "Đang dừng · " + job["task"]
        process = job["process"]
    if process is None:
        return snapshot()
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in descendants:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    return snapshot()


def snapshot_without_lock(job):
    if job is None:
        return "Chưa có tác vụ đang chạy.", "", ""
    log = job["folder"] / "console.txt"
    return job["status"], log.read_text(encoding="utf-8", errors="replace")[-24000:] if log.exists() else "", str(job["folder"])
