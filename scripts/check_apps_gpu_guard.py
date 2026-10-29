from types import SimpleNamespace
from unittest.mock import patch

from apps_ui import gpu_busy


def check():
    cases = [
        ("training", "python.exe", ["python.exe", "-B", "D:/a/apps_model_runner.py"], True),
        ("reward_wrapper", "python.exe", ["python.exe", "scripts/apps_reward_trial.py"], True),
        ("seed_wrapper", "python.exe", ["python.exe", "scripts/apps_reward_seed_probe.py"], True),
        ("adversarial_review", "python.exe", ["python.exe", "scripts/apps_adversarial_review.py"], True),
        ("lora_trial", "python.exe", ["python.exe", "scripts/apps_lora_trial.py"], True),
        ("completion_wrapper", "python.exe", ["python.exe", "C:/a/finish_apps_reward_trial.py"], False),
        ("prefixed_script", "python.exe", ["python.exe", "C:/a/archive_apps_model_runner.py"], False),
        ("dashboard", "python.exe", ["python.exe", "scripts/ui_dashboard.py"], False),
        ("unreadable_command", "python.exe", None, False),
    ]
    for name, binary, command, expected in cases:
        process = SimpleNamespace(pid=999999, info={"name": binary, "cmdline": command})
        with patch("psutil.process_iter", return_value=[process]):
            actual = gpu_busy()
        if actual is not expected:
            raise AssertionError((name, actual, expected))
    return {"status": "passed", "scenarios": len(cases), "exact_script_name_required": True}


if __name__ == "__main__":
    print(check())
