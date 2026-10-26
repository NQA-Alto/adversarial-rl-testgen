import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import ui_demo
from build_demo_catalog import metrics
from dataset_tasks import canonical, compile_trusted
from export_tensorboard import scalar_rows
from probe_round1_learning import score
from restricted_python import RestrictedFunction


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = ui_demo.catalog()
    for path, expected in data["sources"].items():
        assert digest(Path(path)) == expected, path
    checked = 0
    grouped = {}
    for key, case in data["cases"].items():
        reference = compile_trusted(case["reference"])
        buggy = RestrictedFunction(case["buggy_code"], case["domain"])
        for seed, variants in data["predictions"][key].items():
            assert set(variants) == set(ui_demo.VARIANTS)
            for variant, record in variants.items():
                assert len(record["samples"]) == 4
                for sample in record["samples"]:
                    actual = score(sample["raw_output"], sample.get("terminated", True), case, reference, buggy)
                    for name in ("valid_format", "correct_expected", "bug_detected", "reward"):
                        assert actual[name] == sample[name], (key, seed, variant, name)
                    for name in ("gold_output", "buggy_output", "buggy_error"):
                        assert canonical(actual.get(name)) == canonical(sample.get(name)), (key, name)
                    checked += 1
                grouped.setdefault((case["subset"], seed, variant), []).append(record)
            markup = ui_demo.saved(key, int(seed))
            assert markup.count('<article class="test-sample ') == 12
    for (subset, seed, variant), records in grouped.items():
        actual = metrics(records)
        expected = data["metrics"][subset][seed][variant]
        assert actual == expected
    for variant, values in data["experiment"]["decoding_seed_mean_rates"].items():
        for name, expected in values.items():
            mean = sum(data["metrics"]["validation"][str(s)][variant][name] for s in (42, 43, 44)) / 3
            assert abs(mean - expected) < 1e-12
    malicious = {"raw_output": '<script>alert(1)</script>', "terminated": True, "valid_format": False, "correct_expected": False, "bug_detected": False}
    escaped = ui_demo.sample_html(malicious, next(iter(data["cases"].values())), 1)
    assert '<script>' not in escaped and '&lt;script&gt;' in escaped
    training = {}
    for variant, value in data["variants"].items():
        folder = Path(value["adapter_path"]).parent
        assert digest(Path(value["adapter_path"]) / "adapter_model.safetensors") == value["adapter_sha256"]
        points = scalar_rows(folder)
        tag = "sft/loss" if variant == "parent" else "rollout/reward_mean"
        count = sum(name == tag for name, _, _ in points)
        assert count == (72 if variant == "parent" else 96), (variant, count)
        training[variant] = {"source": str(folder), "primary_curve_points": count}
    result = {"status": "passed", "at": datetime.now().astimezone().isoformat(), "saved_outputs_rescored": checked,
              "all_four_outputs_visible": True, "metrics_match_original_logs": True, "source_and_adapter_hashes_match": True,
              "raw_output_html_escaped": True, "training_curves": training, "test_set_model_evaluation": False}
    if args.live:
        launch = ui_demo.read(args.live / "request.json")
        live_catalog = ui_demo.read(args.live / "catalog.json")
        key, seed = launch["case_key"], launch["seed"]
        case = live_catalog["cases"][key]
        reference, buggy = compile_trusted(case["reference"]), RestrictedFunction(case["buggy_code"], case["domain"])
        live = {}
        for variant in ui_demo.VARIANTS:
            record = ui_demo.read(args.live / variant / "prediction.json")
            proof = ui_demo.read(args.live / variant / "result.json")
            assert proof["status"] == "completed" and proof["training"] is False
            expected = live_catalog["predictions"][key].get(str(seed), {}).get(variant)
            for sample in record["samples"]:
                actual = score(sample["raw_output"], sample["terminated"], case, reference, buggy)
                for name in ("valid_format", "correct_expected", "bug_detected", "reward"):
                    assert sample[name] == actual[name]
            live[variant] = {"tests": len(record["samples"]), "hits": sum(s["bug_detected"] for s in record["samples"]),
                "matches_saved_raw_outputs": [s["raw_output"] for s in record["samples"]] == [s["raw_output"] for s in expected["samples"]] if expected else None,
                "peak_torch_allocated_gib": proof["peak_torch_allocated_gib"]}
        result["live_demo"] = {"path": str(args.live), "case_key": key, "seed": seed, "results": live}
    assert checked == 792
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
