import hashlib
import json
import random
from pathlib import Path


def make_schedule(case_ids, focus_ids, groups, seed):
    case_ids, focus_ids = sorted(case_ids), sorted(focus_ids)
    if len(case_ids) != len(set(case_ids)) or len(focus_ids) != len(set(focus_ids)):
        raise ValueError("Duplicate case ID")
    if not focus_ids or not set(focus_ids) < set(case_ids) or groups < 2 or groups % 2:
        raise ValueError("Need distinct Train focus and background cases and an even group count")
    rng = random.Random(seed)
    background = sorted(set(case_ids) - set(focus_ids))
    def draw(pool, count):
        output = []
        while len(output) < count:
            cycle = list(pool)
            rng.shuffle(cycle)
            output.extend(cycle[:count - len(output)])
        return output
    focused, ordinary = draw(focus_ids, groups // 2), draw(background, groups // 2)
    ordered = []
    for pair in zip(focused, ordinary, strict=True):
        pair = list(pair)
        rng.shuffle(pair)
        ordered.extend(pair)
    return {"version": "balanced-v1", "seed": seed, "case_ids": ordered,
            "focus_case_ids": focus_ids, "focus_groups": groups // 2,
            "background_groups": groups // 2}


def load_schedule(path, expected_hash, cases, groups, seed):
    path = Path(path)
    if not expected_hash or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Schedule hash mismatch")
    plan = json.loads(path.read_text(encoding="utf-8"))
    expected = make_schedule([case["case_id"] for case in cases], plan["focus_case_ids"], groups, seed)
    if plan != expected:
        raise ValueError("Schedule differs from its deterministic Train-only design")
    return plan
