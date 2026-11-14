import functools
import hashlib
import html
import json
from pathlib import Path
from build_demo_catalog import NAMES, VARIANTS as VARIANT_LABELS


PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ("parent", "control", "adversarial")
EXAMPLES = [
    ("List reversal · improvement after RL", "validation|p17-reverse/m1", 42),
    ("Palindrome · adversarial arm detects the fault", "validation|p21-palindrome/m1", 44),
    ("Deduplication · regression after RL", "validation|p16-unique-sorted/m1", 42),
    ("Palindrome · all three miss the fault", "validation|p21-palindrome/m2", 42),
]
LABELS = {"bug_detected": "Bug detected", "correct_but_no_bug": "Correct, no bug detected", "wrong_expected": "Wrong expected output",
          "invalid_json": "Invalid JSON", "invalid_schema": "Invalid JSON schema", "input_outside_domain": "Input outside domain", "truncated": "Truncated generation"}


def esc(value):
    return html.escape(str(value))


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=2)
def _load(path, sha):
    target = Path(path)
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Demo data has changed. Rebuild the catalog from the original logs.")
    data = json.loads(raw)
    for case in data["cases"].values():
        case["title"], case["display_specification"] = NAMES[case["problem_id"]]
    for key, item in data["variants"].items():
        item["label"] = VARIANT_LABELS[key]
    return data


def catalog():
    pointer = read(PROJECT / "DEMO_CATALOG.json")
    return _load(pointer["path"], pointer["sha256"])


def choices():
    data = catalog()
    preferred = [key for _, key, _ in EXAMPLES]
    keys = preferred + [key for key in data["cases"] if key not in preferred]
    return [(data["cases"][key]["title"] + " · " + data["cases"][key]["case_id"].split("/")[-1] + (" · Adversarial Train" if key.startswith("pool|") else ""), key) for key in keys]


def domain_text(domain):
    result = []
    for name, spec in domain.items():
        kind = spec["type"]
        if kind == "int":
            value = f'integer from {spec["min"]} to {spec["max"]}'
        elif kind == "list":
            value = f'list of length {spec["min_length"]}–{spec["max_length"]}, elements from {compact(spec["alphabet"])}'
        else:
            value = f'string of length {spec["min_length"]}–{spec["max_length"]}, characters from {compact(spec["alphabet"])}'
        result.append(f'<li><code>{esc(name)}</code>: {esc(value)}</li>')
    return "".join(result)


def context(key):
    case = catalog()["cases"][key]
    subset = "Validation · not used for weight updates" if case["subset"] == "validation" else "Train · adversary-generated code"
    return f'<section class="demo-context"><div><div class="eyebrow">{subset}</div><h2>{esc(case["title"])}</h2><p>{esc(case["display_specification"])}</p><ul>{domain_text(case["domain"])}</ul></div><div><div class="section-label">Code under test</div><pre><code>{esc(case["buggy_code"])}</code></pre></div></section>'


def category(sample, case):
    if "failure_category" in sample:
        return sample["failure_category"]
    if not sample.get("terminated", True):
        return "truncated"
    try:
        value = json.loads(sample["raw_output"])
    except ValueError:
        return "invalid_json"
    if not isinstance(value, dict) or set(value) != {"input", "expected"}:
        return "invalid_schema"
    if not sample["valid_format"]:
        return "input_outside_domain"
    if not sample["correct_expected"]:
        return "wrong_expected"
    return "bug_detected" if sample["bug_detected"] else "correct_but_no_bug"


def sample_html(sample, case, index):
    reason = category(sample, case)
    tone = "hit" if reason == "bug_detected" else "miss" if reason == "correct_but_no_bug" else "bad"
    test = sample.get("test")
    if test is None:
        try:
            test = json.loads(sample["raw_output"])
        except ValueError:
            test = None
    body = ""
    if isinstance(test, dict) and set(test) == {"input", "expected"}:
        body = f'<div class="test-input"><span>Input</span><code>{esc(compact(test["input"]))}</code></div><div class="test-values"><span>Model expected output</span><code>{esc(compact(test["expected"]))}</code></div>'
        if sample.get("valid_format"):
            body += f'<div class="test-values"><span>Reference output</span><code>{esc(compact(sample.get("gold_output")))}</code></div>'
            if "buggy_output" in sample or "buggy_error" in sample:
                value = sample.get("buggy_error") or compact(sample.get("buggy_output"))
                body += f'<div class="test-values"><span>Actual output</span><code>{esc(value)}</code></div>'
    else:
        body = '<div class="invalid-copy">No complete input/expected pair.</div>'
    raw = esc(sample["raw_output"])
    return f'<article class="test-sample {tone}"><header><span>Test {index}</span><strong>{LABELS[reason]}</strong></header>{body}<details><summary>Raw output</summary><pre>{raw}</pre></details></article>'


def render_records(key, records, seed, mode, stamp=""):
    data = catalog()
    case = data["cases"][key]
    cards = ""
    for variant in VARIANTS:
        record = records.get(variant)
        label = data["variants"][variant]["label"]
        if not record:
            cards += f'<section class="model-tests"><h3>{label}</h3><p class="empty-copy">No results available.</p></section>'
            continue
        samples = record["samples"]
        valid = sum(bool(s["valid_format"]) for s in samples)
        correct = sum(bool(s["correct_expected"]) for s in samples)
        hits = sum(bool(s["bug_detected"]) for s in samples)
        cards += f'<section class="model-tests"><div class="model-tests-head"><h3>{label}</h3><strong class="{"hit-text" if hits else "muted-text"}">{hits}/4 bug-detecting tests</strong></div><p class="model-counts">{valid}/4 valid · {correct}/4 correct expected outputs</p>' + "".join(sample_html(s, case, i) for i, s in enumerate(samples, 1)) + '</section>'
    return f'<div class="demo-provenance"><strong>{esc(mode)}</strong><span>Seed {seed} · 4 test / model{(" · " + esc(stamp)) if stamp else ""}</span></div><div class="model-grid">{cards}</div>'


def saved(key, seed):
    data = catalog()
    values = data["predictions"][key]
    if str(seed) not in values:
        return '<div class="empty-copy">Adversarial Train has saved records for seed 42 only. Select seed 42 or run live inference.</div>'
    return render_records(key, values[str(seed)], seed, "SAVED RECORDS")


def comparison(selection):
    data = catalog()
    mean = selection == "Mean across 3 seeds"
    metrics = data["experiment"]["decoding_seed_mean_rates"] if mean else data["metrics"]["validation"][str(selection)]
    definitions = [("Valid tests", "valid_rate", "valid_tests", "generated_tests"), ("Correct expected outputs", "io_accuracy", "correct_expected", "generated_tests"),
                   ("Bug-detecting tests", "attack_rate", "bug_detecting_tests", "generated_tests"), ("Faulty variants detected @4", "empirical_case_detection_at_4", "cases_detected_with_up_to_4_tests", "evaluated_cases")]
    body = ""
    for title, rate, numerator, denominator in definitions:
        body += '<tr><th>' + title + '</th>'
        for variant in VARIANTS:
            value = metrics[variant][rate] * 100
            count = "Mean across 3 seeds" if mean else f'{metrics[variant][numerator]}/{metrics[variant][denominator]}'
            body += f'<td><strong>{value:.2f}%</strong><small>{count}</small><div class="mini-track"><b style="width:{value:.4f}%"></b></div></td>'
        body += '</tr>'
    return '<section class="comparison"><div class="comparison-heading"><h2>Validation comparison</h2><span>6 problems · 18 faulty variants</span></div><div class="table-scroll"><table><thead><tr><th>Metric</th><th>After SFT</th><th>Fixed GRPO</th><th>Adversarial GRPO</th></tr></thead><tbody>' + body + '</tbody></table></div><p class="comparison-meta">Each seed: 72 tests per model. The first three rates include all generated tests, including invalid outputs. @4 uses up to four tests per faulty variant.</p><p class="result-limit">One training seed; three generation seeds (42, 43, 44). Results cover a small internal dataset; Test has not been evaluated for this study.</p></section>'


def training_stability():
    pointer = PROJECT / "LATEST_GRPO_REPLICATES.json"
    if not pointer.is_file():
        return '<p class="empty-copy">The training seed experiment is incomplete. Check Experiments for the active run.</p>'
    metadata = read(pointer)
    root = Path(metadata["path"]).resolve()
    evidence = Path(r"D:\DA\TTNT_De19_Assets\evidence").resolve()
    if not root.is_relative_to(evidence):
        raise ValueError("Invalid replication result path")
    result = read(root / "result.json")
    if result.get("status") != "grpo_training_seed_replicates_completed":
        return '<p class="empty-copy">The experiment is incomplete.</p>'
    lines = ""
    for seed in (42, 43, 44):
        values = result["all_metrics"][str(seed)]
        a, b = (values[arm]["cases_detected_with_up_to_4_tests"] for arm in ("control", "adversarial"))
        lines += f'<tr><th>{seed}</th><td><strong>{a}/18</strong></td><td><strong>{b}/18</strong></td><td>{b-a:+d} code</td></tr>'
    means = result["aggregate"]
    a, b = (means[arm]["empirical_case_detection_at_4"] for arm in ("control", "adversarial"))
    lines += f'<tr><th>Mean ± SD</th><td><strong>{100*a["mean"]:.2f}%</strong><small>± {100*a["sample_std"]:.2f} pp</small></td><td><strong>{100*b["mean"]:.2f}%</strong><small>± {100*b["sample_std"]:.2f} pp</small></td><td>{100*(b["mean"]-a["mean"]):+.2f} pp</td></tr>'
    return '<section class="comparison"><div class="comparison-heading"><h2>GRPO training seed sensitivity</h2><span>Generation seed fixed at 42</span></div><div class="table-scroll"><table><thead><tr><th>Training seed</th><th>Fixed GRPO</th><th>Adversarial GRPO</th><th>Difference</th></tr></thead><tbody>' + lines + '</tbody></table></div><p class="comparison-meta">Faulty cases detected with up to four tests on the same 18 Validation cases. SD is the standard deviation across training seeds, not a confidence interval.</p><p class="result-limit">Same SFT checkpoint and adversarial pool; each arm runs 96 steps with a fresh optimizer. This measures GRPO seed sensitivity, not three independent SFT + adversary + GRPO runs. Demo models use training seed 42.</p></section>'


def live_view(session, active=False):
    if session is None or not (session / "request.json").is_file():
        return '<div class="empty-copy">Select a problem and click Run three models. The base model loads once; the three adapters generate tests in sequence.</div>'
    request = read(session / "request.json")
    records = {}
    for variant in VARIANTS:
        path = session / variant / "prediction.json"
        if path.is_file():
            records[variant] = read(path)
    result_path = session / "result.json"
    result = read(result_path) if result_path.exists() else {}
    state = {"completed": "Completed", "failed": "Error; open Detailed log", "started": "Loading / generating tests" if active else "Run incomplete"}.get(result.get("status"), "Processing" if active else "Run incomplete")
    if (session / "stopped.json").is_file():
        state = "Stopped; only saved results are retained"
    if result.get("error_category") == "memory":
        state = "Out of memory. Close other applications and retry; saved results are retained."
    stamp = result.get("finished_at", request["created_at"])
    return f'<p class="live-state">{state} · {len(records)}/3 model</p>' + render_records(request["case_key"], records, request["seed"], "LIVE LOCAL INFERENCE", stamp)
