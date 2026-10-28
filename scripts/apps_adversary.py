import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from apps_core import ASSETS, all_inputs, canonical, digest, domain_text, load_cases, read, score, write
from apps_model_runner import load_model, run_folder
from audit_apps_run import audit
from restricted_python import RestrictedCodeError, RestrictedFunction


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def observed(function, inputs):
    try:
        return canonical({"output": function(**inputs)})
    except RestrictedCodeError:
        raise
    except (ArithmeticError, IndexError, TypeError, ValueError) as error:
        return canonical({"exception": type(error).__name__})


def signature(outputs):
    return hashlib.sha256(canonical(outputs).encode()).hexdigest()


def contexts(probe, parent):
    audit(probe)
    result = read(probe / "result.json")
    cases, manifest = load_cases("train")
    if result["split"] != "train" or not result["complete_split"] or result["prompt_version"] != "apps-call-v2":
        raise ValueError("A complete Train probe with the fixed v2 prompt is required")
    if result["adapter_sha256"] != digest(parent / "adapter_model.safetensors"):
        raise ValueError("Probe does not evaluate the declared current policy")
    predictions = {row["case_id"]: row for row in rows(probe / "predictions.jsonl")}
    output = {}
    for case in cases:
        identity = str(case["source_id"])
        if identity not in output:
            inputs = list(all_inputs(case))
            reference = RestrictedFunction(case["reference"], case["arguments"])
            output[identity] = {"case": case, "inputs": inputs,
                                "gold": [observed(reference, item) for item in inputs],
                                "positions": {canonical(item): i for i, item in enumerate(inputs)},
                                "known": set(), "suites": {}}
        context = output[identity]
        buggy = RestrictedFunction(case["buggy_code"], case["arguments"])
        context["known"].add(signature([observed(buggy, item) for item in context["inputs"]]))
        samples = predictions[case["case_id"]]["samples"]
        if any(sample["bug_detected"] for sample in samples):
            suite = {canonical(sample["test"]["input"]): sample["test"] for sample in samples if sample["correct_expected"]}
            if suite:
                context["suites"][case["case_id"]] = list(suite.values())
    return {key: value for key, value in output.items() if value["suites"]}, cases, manifest


def assess(source, context, seen):
    try:
        function = RestrictedFunction(source, context["case"]["arguments"])
        outputs = []
        started = time.monotonic()
        for inputs in context["inputs"]:
            if time.monotonic() - started > 5:
                raise RestrictedCodeError("Candidate verification time limit")
            outputs.append(observed(function, inputs))
    except (RestrictedCodeError, NameError, KeyError, SyntaxError) as error:
        return {"accepted": False, "reason": "unsupported_code", "detail": str(error)}
    wrong = [i for i, (actual, gold) in enumerate(zip(outputs, context["gold"], strict=True)) if actual != gold]
    if not wrong:
        return {"accepted": False, "reason": "correct_on_declared_domain"}
    if len(wrong) == len(outputs):
        return {"accepted": False, "reason": "wrong_on_every_input"}
    passed = [identity for identity, suite in context["suites"].items() if suite and all(
        outputs[context["positions"][canonical(test["input"])]] == canonical({"output": test["expected"]}) for test in suite)]
    if not passed:
        return {"accepted": False, "reason": "caught_by_current_policy"}
    behavior = signature(outputs)
    if behavior in context["known"] or behavior in seen:
        return {"accepted": False, "reason": "duplicate_behavior", "behavior_sha256": behavior}
    first = wrong[0]
    return {"accepted": True, "reason": "incorrect_but_passes_current_policy",
            "behavior_sha256": behavior, "passes_policy_cases": passed,
            "domain_inputs": len(outputs), "wrong_inputs": len(wrong),
            "witness_for_audit_only": {"input": context["inputs"][first],
                                       "gold": context["gold"][first], "observed": outputs[first]}}


def code_prompt(case):
    return [{"role": "system", "content": "Solve the programming task correctly. Return only one Python function named solve with the exact argument names. No explanations, tests, imports, helper functions, recursion, while loops or file operations. No type annotations. Use basic arithmetic, comparisons, if/else, bounded for loops and list comprehensions. Do not mutate input objects or lists."},
            {"role": "user", "content": "PROBLEM\n" + case["specification"]
             + "\n\nINPUT CONTRACT FOR THIS EXPERIMENT (overrides larger ranges above)\n" + domain_text(case)
             + "\nSignature: def solve(" + ", ".join(case["arguments"]) + "):"
             + "\nSupported built-ins: abs, min, max, sum, range, len, sorted, set, list, tuple, reversed, enumerate, int, bool, any, all. Supported string/list methods: count, lower, split, find, rfind. Use list comprehensions inside any/all. Output the complete function."}]


def extract_code(raw):
    text = raw.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:python)?\s*\n([\s\S]*?)\n```", text)
        if not match:
            raise RestrictedCodeError("Expected a single code block")
        text = match.group(1).strip()
    return text + "\n"


def audit_pool(folder):
    result = read(folder / "result.json")
    context_by_id, _, _ = contexts(Path(result["probe"]), Path(result["parent_adapter"]))
    if digest(Path(result["probe"]) / "predictions.jsonl") != result["probe_predictions_sha256"]:
        raise ValueError("Policy probe changed")
    events = rows(folder / "candidates.jsonl")
    accepted = rows(folder / "adversarial-cases.jsonl")
    expected = []
    seen = {key: set() for key in context_by_id}
    counts = Counter()
    for event in events:
        context = context_by_id[event["source_id"]]
        if not event["terminated"]:
            verdict = {"accepted": False, "reason": "truncated"}
        else:
            try:
                code = extract_code(event["raw_output"])
                verdict = assess(code, context, seen[event["source_id"]])
            except RestrictedCodeError as error:
                verdict = {"accepted": False, "reason": "unsupported_code", "detail": str(error)}
        if canonical(verdict) != canonical(event["verdict"]):
            raise ValueError("Adversary gate recomputation mismatch")
        counts[verdict["reason"]] += 1
        if verdict["accepted"]:
            seen[event["source_id"]].add(verdict["behavior_sha256"])
            expected.append((event["candidate_id"], code, verdict))
    if [(row["candidate_id"], row["buggy_code"], row["gate"]) for row in accepted] != expected:
        raise ValueError("Accepted pool does not match sampled and audited code")
    if len(events) != result["sampled_candidates"] or dict(counts) != result["gate_counts"] or len(accepted) != result["accepted_candidates"]:
        raise ValueError("Adversary counts mismatch")
    review = {"status": "passed", "candidates_rechecked": len(events), "accepted_rechecked": len(accepted),
              "pool_sha256": digest(folder / "adversarial-cases.jsonl"), "train_only": True, "model_test_evaluation": False}
    write(folder / "review.json", review)
    return review


def generate(args):
    import torch
    os.environ.update(HF_HUB_OFFLINE="1", HF_HOME=str(ASSETS / "cache/huggingface"),
                      TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"), TOKENIZERS_PARALLELISM="false")
    folder = run_folder("adversary")
    print("EVIDENCE:", folder, flush=True)
    context_by_id, cases, manifest = contexts(args.probe, args.parent_adapter)
    result = {"status": "running", "dataset_id": manifest["dataset_id"], "round": args.round,
              "probe": str(args.probe), "probe_predictions_sha256": digest(args.probe / "predictions.jsonl"),
              "parent_adapter": str(args.parent_adapter), "parent_adapter_sha256": digest(args.parent_adapter / "adapter_model.safetensors"),
              "eligible_problems": len(context_by_id), "eligible_case_slots": sum(len(c["suites"]) for c in context_by_id.values()),
              "configuration": {"mode": "sample_correct_solutions_then_filter", "model": "frozen original local Qwen3.5-2B",
                                "samples_per_problem": args.samples_per_problem, "max_new_tokens": 256,
                                "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 42},
              "training": False, "model_test_evaluation": False}
    write(folder / "result.json", result)
    (folder / "source").mkdir()
    for name in ("apps_adversary.py", "apps_core.py", "restricted_python.py", "apps_model_runner.py", "audit_apps_run.py", "apps_curriculum.py"):
        shutil.copy2(Path(__file__).with_name(name), folder / "source" / name)
    write(folder / "policy-suites.json", {key: context["suites"] for key, context in context_by_id.items()})
    for name in ("candidates.jsonl", "adversarial-cases.jsonl"):
        (folder / name).write_text("", encoding="utf-8")
    if not context_by_id:
        raise ValueError("No original Train cases caught by the current policy")
    model, tokenizer, eos = load_model()
    accepted, counts, sampled = [], Counter(), 0
    started = time.perf_counter()
    for identity, context in sorted(context_by_id.items()):
        prompt = tokenizer.apply_chat_template(code_prompt(context["case"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        if inputs.input_ids.shape[1] > 1536:
            raise ValueError("Adversary prompt exceeds budget")
        seen = set()
        for offset in range(0, args.samples_per_problem, 4):
            seed = (42 + int(hashlib.sha256(f'{args.round}/{identity}/{offset}'.encode()).hexdigest()[:8], 16)) % 2**32
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = model.generate(**inputs, num_return_sequences=min(4, args.samples_per_problem - offset),
                                        max_new_tokens=256, max_time=120, do_sample=True, temperature=1.0,
                                        top_p=0.95, top_k=20, eos_token_id=eos,
                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
            for index, tokens in enumerate(output[:, inputs.input_ids.shape[1]:].tolist()):
                end = next((i for i, token in enumerate(tokens) if token in eos), None)
                raw = tokenizer.decode(tokens[:end + 1] if end is not None else tokens, skip_special_tokens=True).strip()
                identity_sample = f'r{args.round}-{identity}-{offset + index + 1:02d}'
                event = {"candidate_id": identity_sample, "source_id": identity, "seed": seed,
                         "batch_position": index, "raw_output": raw, "terminated": end is not None,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                if end is None:
                    verdict = {"accepted": False, "reason": "truncated"}
                else:
                    try:
                        code = extract_code(raw)
                        verdict = assess(code, context, seen)
                    except RestrictedCodeError as error:
                        verdict = {"accepted": False, "reason": "unsupported_code", "detail": str(error)}
                event["verdict"] = verdict
                counts[verdict["reason"]] += 1
                sampled += 1
                with (folder / "candidates.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(canonical(event) + "\n")
                if verdict["accepted"]:
                    seen.add(verdict["behavior_sha256"])
                    row = {"candidate_id": identity_sample, "source_id": identity, "buggy_code": code,
                           "gate": verdict, "sample_source": "local_base_model", "round": args.round}
                    accepted.append(row)
                    with (folder / "adversarial-cases.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(canonical(row) + "\n")
                print("CANDIDATE", sampled, identity_sample, verdict["reason"], flush=True)
            del output
        del inputs
    result.update(status="completed", sampled_candidates=sampled, accepted_candidates=len(accepted),
                  gate_counts=dict(counts), elapsed_seconds=round(time.perf_counter() - started, 3),
                  pool_sha256=digest(folder / "adversarial-cases.jsonl"), peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
    write(folder / "result.json", result)
    audit_pool(folder)
    print("ADVERSARY COMPLETED:", canonical(result), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--parent-adapter", type=Path)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--samples-per-problem", type=int, default=16)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(canonical(audit_pool(args.audit)))
    else:
        if not args.probe or not args.parent_adapter or not 1 <= args.samples_per_problem <= 32:
            parser.error("Provide probe, parent adapter and 1..32 samples per problem")
        generate(args)
