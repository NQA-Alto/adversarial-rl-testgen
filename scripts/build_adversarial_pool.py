import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from check_local_model import Tee
from dataset_tasks import all_inputs, canonical, compile_trusted, input_is_valid, readable_domain
from evaluate_model import summarize
from restricted_python import RestrictedCodeError, RestrictedFunction


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def observed(function, inputs):
    try:
        return canonical({"output": function(**inputs)})
    except RestrictedCodeError:
        raise
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError, NameError) as error:
        return canonical({"exception": type(error).__name__})


def behavior_signature(outputs):
    return hashlib.sha256(canonical(outputs).encode("utf-8")).hexdigest()


def assess_candidate(source, context, seen):
    try:
        function = RestrictedFunction(source, context["task"]["domain"])
        outputs = []
        started = time.monotonic()
        for inputs in context["inputs"]:
            if time.monotonic() - started > 5:
                raise RestrictedCodeError("Candidate verification time limit")
            outputs.append(observed(function, inputs))
    except RestrictedCodeError as error:
        return {"accepted": False, "reason": "restricted_python_rejected", "detail": str(error)}
    wrong = [index for index, (output, gold) in enumerate(zip(outputs, context["gold"], strict=True)) if output != gold]
    if not wrong:
        return {"accepted": False, "reason": "correct_on_finite_domain"}
    passed = []
    for identity, suite in context["policy_suites"].items():
        if suite and all(outputs[context["input_positions"][canonical(test["input"])]] == canonical({"output": test["expected"]}) for test in suite):
            passed.append(identity)
    if not passed:
        return {"accepted": False, "reason": "caught_by_policy_or_no_valid_policy_tests", "wrong_input_count": len(wrong)}
    signature = behavior_signature(outputs)
    if signature in context["known_signatures"] or signature in seen:
        return {"accepted": False, "reason": "behavior_duplicate", "behavior_sha256": signature}
    first = wrong[0]
    return {"accepted": True, "reason": "incorrect_but_passes_current_policy_tests", "behavior_sha256": signature,
            "passes_policy_suites_for_cases": passed, "domain_input_count": len(outputs), "wrong_input_count": len(wrong),
            "witness_for_audit_only": {"input": context["inputs"][first], "gold": json.loads(context["gold"][first]), "observed": json.loads(outputs[first])}}


def prepare_context(args):
    evidence = (args.assets / "evidence").resolve()
    diagnostic = args.diagnostic_run.resolve()
    if not diagnostic.is_relative_to(evidence):
        raise ValueError("Select a diagnostic run inside this project's evidence directory")
    result = read_json(diagnostic / "result.json")
    if result["status"] != "train_diagnostic_completed" or result["preflight_only"]:
        raise ValueError("Expected a complete real Train diagnostic")
    training_path = Path(result["training_run"]) / "result.json"
    if digest(training_path) != result["frozen_training_result_sha256"]:
        raise ValueError("Referenced training result changed")
    training = read_json(training_path)
    adapter = Path(result["adapter_path"])
    if training["status"] != "static_grpo_completed" or digest(adapter / "adapter_model.safetensors") != result["adapter_sha256"]:
        raise ValueError("Missing verified protagonist training or changed adapter")
    source = Path(__file__).with_name("dataset_tasks.py")
    if digest(source) != training["source_hashes"]["dataset_tasks.py"]:
        raise ValueError("Frozen task/verifier source changed")
    dataset = args.assets / "data/internal-v1"
    manifest = read_json(dataset / "manifest.json")
    if digest(dataset / "train.jsonl") != manifest["files"]["train.jsonl"] or manifest["files"]["train.jsonl"] != training["train_split_sha256"]:
        raise ValueError("Frozen Train data changed")
    cases = read_rows(dataset / "train.jsonl")
    predictions = read_rows(diagnostic / "trained/predictions.jsonl")
    evaluation = read_json(diagnostic / "trained/result.json")
    if evaluation["split"] != "train" or evaluation["status"] != "evaluation_completed" or evaluation["model_variant"] != "base_plus_adapter":
        raise ValueError("Policy tests must come from trained model on Train only")
    if evaluation["adapter_sha256"] != result["adapter_sha256"] or evaluation["task_source_sha256"] != digest(source):
        raise ValueError("Protagonist test provenance differs")
    if evaluation["dataset_split_sha256"] != training["train_split_sha256"] or evaluation["metrics"] != summarize(predictions, True, len(cases)):
        raise ValueError("Policy prediction records or data differ from saved metrics")
    if len(cases) != 36 or len(predictions) != 36 or {row["case_id"] for row in predictions} != {row["case_id"] for row in cases}:
        raise ValueError("Expected all 36 frozen Train cases")
    case_lookup = {case["case_id"]: case for case in cases}
    contexts = {}
    for case in cases:
        identity = case["problem_id"]
        if identity not in contexts:
            reference = compile_trusted(case["reference"])
            inputs = list(all_inputs(case["domain"]))
            gold = [canonical({"output": reference(**item)}) for item in inputs]
            contexts[identity] = {"task": case, "inputs": inputs, "gold": gold,
                                  "input_positions": {canonical(item): index for index, item in enumerate(inputs)},
                                  "policy_suites": {}, "known_signatures": set()}
        context = contexts[identity]
        existing = compile_trusted(case["buggy_code"])
        context["known_signatures"].add(behavior_signature([observed(existing, item) for item in context["inputs"]]))
    for record in predictions:
        case = case_lookup[record["case_id"]]
        context = contexts[case["problem_id"]]
        suite = {}
        for sample in record["samples"]:
            if not (sample["terminated"] and sample["valid_format"] and sample["correct_expected"]):
                continue
            test = json.loads(sample["raw_output"])
            if set(test) != {"input", "expected"} or not input_is_valid(test["input"], case["domain"]):
                raise ValueError("Claimed valid policy test is invalid")
            position = context["input_positions"][canonical(test["input"])]
            if canonical({"output": test["expected"]}) != context["gold"][position]:
                raise ValueError("Policy test expected value is incorrect")
            suite[canonical(test["input"])] = test
        context["policy_suites"][case["case_id"]] = list(suite.values())
    if len(contexts) != 12:
        raise ValueError("Expected 12 Train problems")
    return contexts, training, result


def code_prompt(task):
    return [
        {"role": "system", "content": "Solve the programming task correctly. Output only one Python function named solve with the exact argument names. No explanation, tests, imports, helper functions, recursion, while loops or file operations. Use basic integer arithmetic, comparisons, if/else, bounded for loops, built-ins and list comprehensions. Use comprehensions instead of append. Do not use type annotations."},
        {"role": "user", "content": "Specification: " + task["specification"] + "\nArguments and legal inputs:\n" + readable_domain(task["domain"]) + "\nReturn only the complete solve function. Supported built-ins: abs, min, max, sum, range, len, sorted, set, list, tuple, reversed, enumerate, int, bool, any, all. No mutation of lists or argument objects."},
    ]


def extract_code(raw):
    text = raw.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:python)?\s*\n([\s\S]*?)\n```", text)
        if not match:
            raise RestrictedCodeError("Expected a single Python code block")
        text = match.group(1).strip()
    return text + "\n"


def run(args, folder, result):
    contexts, training, diagnostic = prepare_context(args)
    from transformers import AutoTokenizer
    model_path = Path(training["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    prompts = {identity: tokenizer.apply_chat_template(code_prompt(context["task"]), tokenize=False, add_generation_prompt=True, enable_thinking=False) for identity, context in sorted(contexts.items())}
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts.values()]
    if max(lengths) > 768:
        raise ValueError("Adversary prompt exceeds the checked token budget")
    result.update(model_path=str(model_path), adversary_model_variant="original_base_frozen", protagonist_training_run=diagnostic["training_run"],
                  protagonist_adapter_sha256=diagnostic["adapter_sha256"], diagnostic_run=str(args.diagnostic_run.resolve()),
                  diagnostic_result_sha256=digest(args.diagnostic_run / "result.json"),
                  dataset_id=training["dataset_id"], train_split_sha256=training["train_split_sha256"],
                  problem_count=len(contexts), planned_candidates=len(contexts) * args.samples_per_problem,
                  policy_cases_with_nonempty_valid_suite=sum(bool(suite) for context in contexts.values() for suite in context["policy_suites"].values()),
                  prompt_token_range=[min(lengths), max(lengths)],
                  configuration={"mode": "unconditional_correct_solution_sampling_then_filtering", "seed": 42,
                                 "samples_per_problem": args.samples_per_problem, "max_new_tokens": 256, "temperature": 1.0,
                                 "top_p": 0.95, "top_k": 20, "thinking": False, "dtype": "bfloat16",
                                 "gate": "At least one nonempty current per-case policy suite passed, at least one finite-domain oracle input failed, behavior distinct from the three curated mutants and previous accepted candidates of the same problem"},
                  limitations=["Internal finite-domain data, not the paper's APPS/Codeforces benchmark", "Restricted Python subset; unsupported programs are rejected rather than called buggy", "Adversary weights are frozen; only the protagonist has been trained with RL", "No new RL update is performed in this command"])
    write_json(folder / "policy-suites.json", {identity: context["policy_suites"] for identity, context in contexts.items()})
    if args.preflight:
        result["status"] = "adversary_preflight_passed_no_model_weights_loaded"
        print("PREFLIGHT PASSED:", len(contexts), "Train problems | planned candidates:", result["planned_candidates"], flush=True)
        print("No model weights loaded, code generated, parameter updates or Test evaluation.", flush=True)
        return
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and BF16 are required")
    result.update(gpu=torch.cuda.get_device_name(0), versions={name: importlib.metadata.version(name) for name in ("torch", "transformers")})
    torch.cuda.reset_peak_memory_stats()
    print("[1/3] Load frozen original local model for adversary; no protagonist adapter", flush=True)
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
    result["loading_info"] = loading
    if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
        raise RuntimeError("Unexpected original model checkpoint loading result")
    model.eval()
    model.requires_grad_(False)
    declared = model.generation_config.eos_token_id
    eos_ids = sorted({tokenizer.eos_token_id, *(declared if isinstance(declared, list) else [declared])} - {None})
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    result["configuration"]["eos_token_ids"] = eos_ids
    accepted, counts = [], Counter()
    destination = folder / "adversarial-cases.jsonl"
    destination.write_text("", encoding="utf-8")
    started = time.perf_counter()
    print("[2/3] Sample solutions, then verify in the restricted interpreter", flush=True)
    for identity, context in sorted(contexts.items()):
        seen = set()
        prompt = prompts[identity]
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        for index in range(args.samples_per_problem):
            seed = (42 + int(hashlib.sha256((identity + "/" + str(index)).encode()).hexdigest()[:8], 16)) % 2**32
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256, max_time=60, do_sample=True, temperature=1.0,
                                        top_p=0.95, top_k=20, use_cache=True, pad_token_id=pad, eos_token_id=eos_ids)
            tokens = output[0, inputs["input_ids"].shape[1]:].tolist()
            end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
            raw = tokenizer.decode(tokens[:end + 1] if end is not None else tokens, skip_special_tokens=True).strip()
            event = {"problem_id": identity, "sample": index + 1, "seed": seed, "raw_output": raw,
                     "terminated": end is not None, "prompt_messages": code_prompt(context["task"]), "sample_source": "local_original_model_generated"}
            if end is None:
                verdict = {"accepted": False, "reason": "truncated_generation"}
            else:
                try:
                    source = extract_code(raw)
                    event["candidate_code"] = source
                    verdict = assess_candidate(source, context, seen)
                except RestrictedCodeError as error:
                    verdict = {"accepted": False, "reason": "restricted_python_rejected", "detail": str(error)}
            event["verdict"] = verdict
            counts[verdict["reason"]] += 1
            if verdict["accepted"]:
                seen.add(verdict["behavior_sha256"])
                task = context["task"]
                case = {key: task[key] for key in ("problem_id", "split_group", "specification", "domain", "reference")}
                case.update(case_id=identity + "/adversarial-" + verdict["behavior_sha256"][:12], split="train", buggy_code=source,
                            origin="local_adversary_unconditional_round1", origin_policy_cases=verdict["passes_policy_suites_for_cases"],
                            behavior_sha256=verdict["behavior_sha256"])
                accepted.append(case)
                with destination.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(case, ensure_ascii=False) + "\n")
            with (folder / "candidates.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(f'CANDIDATE {sum(counts.values())}/{result["planned_candidates"]} | {identity} | {verdict["reason"]}', flush=True)
            result.update(generated_candidates=sum(counts.values()), accepted_candidates=len(accepted), rejection_and_acceptance_counts=dict(counts))
            write_json(folder / "partial-result.json", result)
    torch.cuda.synchronize()
    result.update(status="adversary_pool_completed" if accepted else "adversary_completed_no_accepted_candidates",
                  generation_and_verification_seconds=round(time.perf_counter() - started, 3),
                  peak_torch_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                  accepted_cases_path=str(destination), accepted_cases_sha256=digest(destination),
                  accepted_problem_count=len({case["problem_id"] for case in accepted}))
    print("[3/3] ADVERSARY SUMMARY:", flush=True)
    print(json.dumps({key: result[key] for key in ("generated_candidates", "accepted_candidates", "accepted_problem_count", "rejection_and_acceptance_counts", "generation_and_verification_seconds", "peak_torch_allocated_gib")}, indent=2), flush=True)
    print("These candidates have not been used for a new RL update. No Test evaluation.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Sample a local frozen adversary and verify novel incorrect code on Train only")
    parser.add_argument("--assets", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets"))
    parser.add_argument("--diagnostic-run", type=Path, default=Path(r"D:\DA\TTNT_De19_Assets\evidence\train-diagnostic-20261025-181148-811724"))
    parser.add_argument("--samples-per-problem", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples_per_problem <= 16:
        parser.error("Samples per problem must be from 1 through 16")
    for key, value in {"HF_HOME": str(args.assets / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "TORCHINDUCTOR_CACHE_DIR": str(args.assets / "cache/torchinductor"),
        "TEMP": str(args.assets / "temp"), "TMP": str(args.assets / "temp")}.items():
        os.environ[key] = value
    folder = args.assets / "evidence" / ("adversary-pool-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / "source"
    source.mkdir()
    for name in (Path(__file__).name, "restricted_python.py", "check_adversary_gate.py", "dataset_tasks.py", "evaluate_model.py", "check_local_model.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    result = {"started_at": datetime.now().astimezone().isoformat(), "status": "started", "training": False, "test_set_evaluation": False,
              "preflight_only": args.preflight, "source_hashes": {path.name: digest(path) for path in source.iterdir()}}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    code = 0
    with (folder / "console.txt").open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        try:
            print("Evidence:", folder, flush=True)
            run(args, folder, result)
            print("STATUS:", result["status"], flush=True)
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            code = 130
            print("Interrupted; partial candidates preserved.", flush=True)
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            code = 1
            traceback.print_exc()
        finally:
            result["finished_at"] = datetime.now().astimezone().isoformat()
            write_json(folder / "result.json", result)
            print("Saved evidence:", folder, flush=True)
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return code


if __name__ == "__main__":
    raise SystemExit(main())
