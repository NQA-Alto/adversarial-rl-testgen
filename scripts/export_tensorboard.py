import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_lines(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def scalar_rows(folder):
    values = []
    history = load_json(folder / "trainer-log-history.json", []) + read_lines(folder / "trainer-metrics.jsonl")
    for row in history:
        step = row.get("step")
        if not isinstance(step, (int, float)):
            continue
        for key in ("reward", "loss", "grad_norm", "learning_rate", "entropy", "frac_reward_zero_std"):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(("trainer/" + key, int(step), float(value)))
    for row in read_lines(folder / "rollouts.jsonl"):
        rewards = row.get("rewards")
        step = row.get("group")
        if rewards and isinstance(step, int) and all(isinstance(value, (int, float)) and math.isfinite(value) for value in rewards):
            values.append(("rollout/reward_mean", step, sum(rewards) / len(rewards)))
            values.append(("rollout/policy_signal", step, float(bool(row.get("policy_signal")))))
            samples = row.get("samples", [])
            for field, tag in (("valid_format", "verified_valid_rate"), ("correct_expected", "verified_io_accuracy"),
                               ("bug_detected", "verified_attack_rate")):
                if samples and all(type(sample.get(field)) is bool for sample in samples):
                    values.append(("rollout/" + tag, step, sum(sample[field] for sample in samples) / len(samples)))
    for row in read_lines(folder / "gradient-audit.jsonl"):
        step, value = row.get("step"), row.get("gradient_norm")
        if isinstance(step, int) and isinstance(value, (int, float)) and math.isfinite(value):
            values.append(("gradient/norm_observed", step, float(value)))
    for row in load_json(folder / "gradients.json", []):
        if isinstance(row.get("step"), int) and isinstance(row.get("norm"), (int, float)):
            values.append(("gradient/norm_observed", row["step"], float(row["norm"])))
    for row in read_lines(folder / "sft-steps.jsonl"):
        step = row.get("step")
        if not isinstance(step, int):
            continue
        for key, tag in (("loss", "sft/loss"), ("gradient_norm", "gradient/norm_observed")):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append((tag, step, float(value)))
    return values


class Exporter:
    def __init__(self, evidence, scope="all"):
        self.evidence = evidence
        self.scope = scope
        self.output = evidence / ("tensorboard-apps" if scope == "apps" else "tensorboard")
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output / "export-state.json"
        self.state = load_json(self.state_file, {})
        self.writers = {}

    def sync(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        added = 0
        folders = {path.parent for name in ("rollouts.jsonl", "trainer-log-history.json", "trainer-metrics.jsonl", "sft-steps.jsonl") for path in self.evidence.rglob(name)}
        for folder in sorted(folders):
            relative = folder.relative_to(self.evidence)
            if self.scope == "apps" and not relative.parts[0].startswith("apps-"):
                continue
            if any(part in ("source", "tensorboard", "ui-jobs") or "self-check" in part or "preflight" in part for part in relative.parts):
                continue
            run = relative.as_posix()
            rows = scalar_rows(folder)
            known = self.state.setdefault(run, {})
            for tag, step, value in rows:
                key = tag + "/" + str(step)
                fingerprint = hashlib.sha256(str(value).encode()).hexdigest()
                if known.get(key) == fingerprint:
                    continue
                if run not in self.writers:
                    self.writers[run] = EventFileWriter(str(self.output / relative))
                summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
                self.writers[run].add_event(Event(wall_time=time.time(), step=step, summary=summary))
                known[key] = fingerprint
                added += 1
        for writer in self.writers.values():
            writer.flush()
        temp = self.state_file.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(temp, self.state_file)
        return added

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--scope", choices=("all", "apps"), default="all")
    args = parser.parse_args()
    exporter = Exporter(args.assets / "evidence", args.scope)
    try:
        while True:
            count = exporter.sync()
            if count:
                print("Exported scalar points:", count, flush=True)
            if not args.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        exporter.close()


if __name__ == "__main__":
    main()
