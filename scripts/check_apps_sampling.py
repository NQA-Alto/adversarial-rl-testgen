import hashlib
import json
import tempfile
from pathlib import Path

from apps_sampling import load_schedule, make_schedule


def main():
    from trl.trainer.utils import RepeatSampler
    cases = [{"case_id": f"train-{i}"} for i in range(145)]
    focus = [case["case_id"] for case in cases[:17]]
    plan = make_schedule([case["case_id"] for case in cases], focus, 64, 42)
    assert plan == make_schedule([case["case_id"] for case in reversed(cases)], list(reversed(focus)), 64, 42)
    assert sum(identity in focus for identity in plan["case_ids"]) == 32
    sampler = RepeatSampler(list(range(64)), mini_repeat_count=4, batch_size=1,
                            repeat_count=4, shuffle=False, seed=42)
    indices = list(sampler)
    assert len(indices) == 64 * 16
    assert [indices[i] for i in range(0, len(indices), 16)] == list(range(64))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "schedule.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert load_schedule(path, digest, cases, 64, 42) == plan
        for wrong_hash, wrong_cases, wrong_steps, wrong_seed in (
            ("bad", cases, 64, 42), (digest, cases[1:], 64, 42),
            (digest, cases, 32, 42), (digest, cases, 64, 43),
        ):
            try:
                load_schedule(path, wrong_hash, wrong_cases, wrong_steps, wrong_seed)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid schedule accepted")
    print("PASS: deterministic 50/50 Train schedule, actual TRL sampler order, tamper and split guards")


if __name__ == "__main__":
    main()
