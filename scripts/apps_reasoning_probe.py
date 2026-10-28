import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from apps_core import ASSETS, canonical, digest, domain_text, load_cases, metrics, read, score, write
from apps_model_runner import PROJECT, load_model, run_folder


SOURCE_IDS = (2667, 2703, 2865, 2899, 2948, 2987, 3019, 3024)


def messages(case):
    return [{"role": "system", "content": "Generate one test for the Python function. Follow the finite input contract exactly. Compute expected from the problem specification, not from the potentially buggy implementation. Put the final answer in exactly one <answer>...</answer> block containing one JSON object with exactly the keys input and expected. No markdown fences, no text after </answer>."},
            {"role": "user", "content": "PROBLEM SPECIFICATION\n" + case["specification"]
             + "\n\nIMPLEMENTATION TO TEST\n" + case["buggy_code"]
             + "\n\nFINITE INPUT CONTRACT (overrides larger ranges and examples above)\n" + domain_text(case)
             + "\nExact argument names: " + canonical(case["arguments"])
             + '\nExample for an unrelated addition function: <answer>{"input":{"a":2,"b":3},"expected":5}</answer>.'
             + "\nChoose concrete allowed inputs that expose a bug, and compute the correct expected value. Do not reuse the example's field names or values. Finish with <answer>{\"input\":{...},\"expected\":...}</answer> with concrete JSON values."}]


def score_completion(raw, terminated, case, thinking):
    if not terminated:
        result = score(raw, False, case)
        result["raw_completion"] = raw
        return result
    body = raw
    reason = ""
    if thinking:
        parts = raw.split("</think>")
        if len(parts) != 2 or not parts[0].strip() or "<think>" in parts[0]:
            result = score("", True, case)
            result.update(raw_completion=raw, error="thinking_wrapper")
            return result
        reason, body = parts
    matched = re.fullmatch(r"\s*<answer>([\s\S]*?)</answer>\s*", body)
    if not matched or "<answer>" in matched.group(1) or "</answer>" in matched.group(1):
        result = score("", True, case)
        result.update(raw_completion=raw, error="answer_wrapper")
        return result
    result = score(matched.group(1).strip(), True, case)
    result.update(raw_completion=raw, reasoning_characters=len(reason), declared_answer_extraction=True)
    return result


def audit(folder):
    plan = read(folder / "plan.json")
    result = read(folder / "result.json")
    incomplete = result["status"] == "stopped_early_infeasible"
    cases, manifest = load_cases("train")
    by_id = {case["case_id"]: case for case in cases}
    if plan["dataset_id"] != manifest["dataset_id"] or plan["train_sha256"] != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Train data changed")
    if digest(Path(plan["adapter"]) / "adapter_model.safetensors") != plan["adapter_sha256"]:
        raise ValueError("Probe adapter changed")
    expected_ids = plan["case_ids"]
    total = 0
    recomputed_metrics = {}
    signatures = {}
    for mode in ("direct", "thinking"):
        path = folder / (mode + "-predictions.jsonl")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        identities = [row["case_id"] for row in records]
        required = expected_ids[:len(identities)] if incomplete and mode == "thinking" else expected_ids
        if identities != required or not identities:
            raise ValueError("Probe subset incomplete")
        for row in records:
            if len(row["samples"]) != 4:
                raise ValueError("Probe test budget changed")
            for sample in row["samples"]:
                checked = score_completion(sample["raw_completion"], sample["terminated"], by_id[row["case_id"]], mode == "thinking")
                for key in ("valid_format", "correct_expected", "bug_detected", "reward", "error"):
                    if sample.get(key) != checked.get(key):
                        raise ValueError("Probe rescoring mismatch: " + key)
                total += 1
        recomputed_metrics[mode] = metrics(records)
        signatures[mode] = digest(path)
    stored = {key: value.get("metrics", value.get("partial_metrics_not_comparable_to_complete_arm")) for key, value in result["modes"].items()}
    if canonical(recomputed_metrics) != canonical(stored):
        raise ValueError("Probe metric mismatch")
    if incomplete:
        partial = result["modes"]["thinking"]
        upper = (partial["terminated_samples"] + partial["planned_samples"] - partial["completed_samples"]) / partial["planned_samples"]
        if upper >= .8 or upper != partial["best_possible_terminated_fraction"]:
            raise ValueError("Futility bound mismatch")
    review = {"status": "passed", "samples_rescored": total, "prediction_hashes": signatures, "train_only": True,
              "training": False, "incomplete_arm": incomplete, "model_test_evaluation": False}
    write(folder / "review.json", review)
    return review


def self_check():
    case = load_cases("train")[0][0]
    arguments = {"distance_to_pump": 2, "mpg": 1, "fuel_left": 2}
    payload = canonical({"input": arguments, "expected": True})
    direct = "<answer>" + payload + "</answer>"
    thought = "Fuel suffices exactly. </think>" + direct
    assert score_completion(direct, True, case, False)["correct_expected"]
    assert score_completion(thought, True, case, True)["correct_expected"]
    for raw, thinking in ((payload, False), (direct + payload, False), (direct, True),
                          ("missing closing think " + direct, True), ("</think>" + direct, True),
                          ("```json\n" + direct + "\n```", False)):
        assert not score_completion(raw, True, case, thinking)["valid_format"]
    assert not score_completion(direct, False, case, False)["valid_format"]
    wrong = "<answer>" + canonical({"input": arguments, "expected": False}) + "</answer>"
    assert not score_completion(wrong, True, case, False)["correct_expected"]
    print("Declared thinking/answer extraction: 10 checks passed, no answer repair")


def run():
    import torch
    from apps_ui import gpu_busy
    if gpu_busy(ignore_pid=os.getpid()):
        raise RuntimeError("Another project GPU job is active")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HOME=str(ASSETS / "cache/huggingface"),
                      TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"), TOKENIZERS_PARALLELISM="false")
    state = read(PROJECT / "APPS_STATE.json")
    adapter = Path(state["sft_training"]["adapter_path"])
    if digest(adapter / "adapter_model.safetensors") != state["sft_training"]["adapter_sha256"]:
        raise ValueError("SFT adapter changed")
    cases, manifest = load_cases("train")
    selected = [min((case for case in cases if case["source_id"] == identity), key=lambda case: case["case_id"]) for identity in SOURCE_IDS]
    folder = run_folder("reasoning-probe")
    print("EVIDENCE:", folder, flush=True)
    source = folder / "source"
    source.mkdir()
    files = ("apps_reasoning_probe.py", "apps_core.py", "apps_model_runner.py", "local_model_loader.py", "restricted_python.py")
    for name in files:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    plan = {"method": "FROZEN_SFT_INFERENCE_ABLATION_NOT_RL", "dataset_id": manifest["dataset_id"],
            "train_sha256": manifest["splits"]["train"]["sha256"], "source_ids": list(SOURCE_IDS),
            "case_ids": [case["case_id"] for case in selected], "selection": "Prespecified eight diverse Train tasks, first mutant per source; no Validation/Test selection",
            "adapter": str(adapter), "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
            "prompt_version": "apps-tagged-probe-v1", "seed": 42, "tests_per_case": 4,
            "configuration": {"max_new_tokens": 768, "max_prompt_tokens": 1536, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
            "comparison": "Same messages, checkpoint, seed and token cap; only native enable_thinking differs. Prefix/answer parsing is declared before generation.",
            "acceptance_rule": "Thinking is a candidate for a later Validation experiment only if >=80% terminated, IO accuracy improves by at least 5 percentage points and usable bug-test count does not fall versus direct.",
            "model_card_note": "The downloaded Qwen3.5-2B model card warns of thinking loops; truncated outputs remain failures.",
            "source_hashes": {name: digest(source / name) for name in files}, "model_test_evaluation": False,
            "training": False, "publish_demo_checkpoint": False}
    write(folder / "plan.json", plan)
    result = {"status": "running", "plan": plan, "modes": {}, "training": False, "model_test_evaluation": False}
    write(folder / "result.json", result)
    model, tokenizer, eos = load_model(adapter)
    started_all = time.perf_counter()
    for mode in ("direct", "thinking"):
        records, lengths, finished = [], [], 0
        started = time.perf_counter()
        for case in selected:
            prompt = tokenizer.apply_chat_template(messages(case), tokenize=False, add_generation_prompt=True, enable_thinking=mode == "thinking")
            if mode == "thinking" and not prompt.rstrip().endswith("<think>"):
                raise ValueError("Native thinking prefix differs from declared parser")
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            if inputs.input_ids.shape[1] > 1536:
                raise ValueError("Probe prompt over budget")
            seed = (42 + int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)) % 2**32
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = model.generate(**inputs, num_return_sequences=4, max_new_tokens=768, max_time=120,
                                        do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                                        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
            samples = []
            for tokens in output[:, inputs.input_ids.shape[1]:].tolist():
                end = next((i for i, token in enumerate(tokens) if token in eos), None)
                actual = tokens[:end + 1] if end is not None else tokens
                raw = tokenizer.decode(tokens[:end] if end is not None else tokens, skip_special_tokens=False).strip()
                checked = score_completion(raw, end is not None, case, mode == "thinking")
                if checked.get("error") == "verifier_limit":
                    raise ValueError("Verifier limit during probe")
                checked["generated_tokens"] = len(actual)
                samples.append(checked)
                lengths.append(len(actual))
                finished += end is not None
            row = {"case_id": case["case_id"], "source_id": case["source_id"], "case_seed": seed,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "message_sha256": hashlib.sha256(canonical(messages(case)).encode()).hexdigest(), "samples": samples}
            records.append(row)
            with (folder / (mode + "-predictions.jsonl")).open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical(row) + "\n")
            print(mode, case["case_id"], "valid", sum(sample["valid_format"] for sample in samples), "expected", sum(sample["correct_expected"] for sample in samples), "hits", sum(sample["bug_detected"] for sample in samples), flush=True)
            del output, inputs
        result["modes"][mode] = {"metrics": metrics(records), "terminated_rate": finished / len(lengths),
                                 "mean_generated_tokens": sum(lengths) / len(lengths), "elapsed_seconds": round(time.perf_counter() - started, 3)}
        write(folder / "result.json", result)
    direct, thinking = result["modes"]["direct"], result["modes"]["thinking"]
    eligible = thinking["terminated_rate"] >= .8 and thinking["metrics"]["io_accuracy"] >= direct["metrics"]["io_accuracy"] + .05 and thinking["metrics"]["bug_detecting_tests"] >= direct["metrics"]["bug_detecting_tests"]
    if digest(adapter / "adapter_model.safetensors") != plan["adapter_sha256"]:
        raise ValueError("Frozen adapter changed")
    result.update(status="completed", thinking_eligible_for_later_validation=eligible,
                  elapsed_seconds=round(time.perf_counter() - started_all, 3),
                  peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), adapter_unchanged=True)
    write(folder / "result.json", result)
    audit(folder)
    print("REASONING PROBE COMPLETED:", canonical(result["modes"]), "eligible", eligible, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.audit:
        print(canonical(audit(args.audit)))
    else:
        run()
