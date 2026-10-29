from unittest.mock import patch

import apps_final_evaluate as final
from apps_core import digest, read, write
from apps_model_runner import run_folder


def check():
    root = run_folder("final-guard-check")
    plan = {"source_hashes": {}}
    write(root / "plan.json", plan)
    plan_hash = digest(root / "plan.json")
    lock = root / "unit-lock.json"
    scenarios = (("completed_event", "completed", False, plan_hash, "Fresh prepared"),
                 ("foreign_lock", "failed", True, "different", "Only the same"),
                 ("changed_protocol", "prepared", False, plan_hash, "Frozen protocol"))
    passed = []
    for name, status, resume, lock_hash, expected in scenarios:
        write(root / "result.json", {"status": status, "plan_sha256": "changed" if name == "changed_protocol" else plan_hash})
        write(lock, {"root": str(root), "plan_sha256": lock_hash})
        with patch.object(final, "LOCK", lock), patch.object(final, "gpu_busy", return_value=False), \
             patch.object(final.subprocess, "check_output", return_value="7000\n"), \
             patch.object(final, "load_cases", side_effect=AssertionError("Holdout opened by a rejected request")) as load, \
             patch.object(final, "verify", side_effect=AssertionError("Invalid request reached source verification")) as verify:
            try:
                final.execute(root, resume=resume)
            except ValueError as error:
                if expected not in str(error):
                    raise
            else:
                raise AssertionError("Unsafe final request was accepted")
            if load.called or verify.called:
                raise AssertionError("Rejected request reached later evaluation stages")
        passed.append(name)
    try:
        final.verify({"source_hashes": {"apps_final_evaluate.py": "wrong"}})
    except ValueError as error:
        if "Frozen evaluation source changed" not in str(error):
            raise
    else:
        raise AssertionError("Changed evaluator source accepted")
    passed.append("changed_source")
    report = {"status": "passed", "scenarios": passed, "models_loaded": False, "holdout_content_read": False,
              "real_final_lock_touched": False, "unit_fixtures_only": True}
    write(root / "review.json", report)
    print(report)


if __name__ == "__main__":
    check()
