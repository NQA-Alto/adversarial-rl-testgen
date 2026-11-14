import html
import json
import re
from datetime import datetime
from pathlib import Path


REASONS = {
    "incorrect_but_passes_current_policy_tests": ("Accepted", "Fault not detected by current tests", "accepted"),
    "correct_on_finite_domain": ("Rejected", "Correct across the finite domain", "correct"),
    "restricted_python_rejected": ("Rejected", "Outside the supported Python subset", "unsupported"),
    "caught_by_policy_or_no_valid_policy_tests": ("Rejected", "Detected or no valid policy tests", "caught"),
    "behavior_duplicate": ("Rejected", "Duplicate behavior", "duplicate"),
    "truncated_generation": ("Rejected", "Incomplete generated code", "unsupported"),
}
PROBLEMS = {
    "p01-nonnegative": "Nonnegative number", "p02-positive": "Positive number", "p03-closed-interval": "Closed interval",
    "p04-even": "Even number", "p06-clamp": "Clamp a value", "p07-sign": "Sign of a number", "p09-factorial": "Factorial",
    "p11-max-pair": "Maximum of two numbers", "p13-count-positive": "Count positive numbers", "p14-count-even": "Count even numbers",
    "p15-sum-positive": "Sum positive numbers", "p20-max-list": "Maximum of a list",
}


def esc(value):
    return html.escape(str(value))


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def rows(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            pass
    return result


def elapsed_text(seconds):
    try:
        seconds = max(0, round(float(seconds)))
    except (ValueError, TypeError):
        return ""
    return f"{seconds // 60} min {seconds % 60:02d} sec" if seconds >= 60 else f"{seconds} sec"


def latest_job(evidence, jobs):
    if jobs.session is not None:
        folder = jobs.session
    else:
        parent = evidence / "ui-jobs"
        choices = sorted((p for p in parent.iterdir() if p.is_dir() and (p / "launch.json").exists()), reverse=True) if parent.exists() else []
        folder = choices[0] if choices else None
    if folder is None:
        return None, {}, ""
    metadata = read_json(folder / "launch.json", {})
    try:
        text = (folder / "console.txt").read_text(encoding="utf-8")
    except OSError:
        text = ""
    return folder, metadata, text


def run_folder(evidence, text):
    from pathlib import Path
    matches = re.findall(r"(?m)^(?:Evidence|Saved evidence):\s*(.+?)\s*$", text)
    for value in reversed(matches):
        folder = Path(value.strip())
        if folder.resolve().is_relative_to(evidence.resolve()) and folder.is_dir():
            return folder
    return None


def workbench(evidence, jobs):
    folder, metadata, log = latest_job(evidence, jobs)
    running = jobs.process is not None and jobs.process.poll() is None
    target = run_folder(evidence, log)
    result = read_json(target / "result.json", {}) if target else {}
    partial = read_json(target / "partial-result.json", {}) if target else {}
    data = result if result.get("status") not in (None, "started") else partial
    status = "Running" if running else "Completed" if data.get("status", "").endswith("completed") or data.get("status") == "adversary_completed_no_accepted_candidates" else "Ready" if folder is None else "Stopped" if jobs.stopped else "Incomplete"
    if jobs.process is not None and jobs.process.poll() not in (None, 0) and not jobs.stopped:
        status = "Error"
    if folder is not None and (folder / "stopped.json").is_file() and not running:
        status = "Stopped"
    task = metadata.get("task", "No runs available")
    stamp = metadata.get("started_at", "")
    try:
        stamp = datetime.fromisoformat(stamp).strftime("%H:%M · %d/%m")
    except ValueError:
        pass
    duration = elapsed_text(data.get("generation_and_verification_seconds") or data.get("training_seconds") or data.get("evaluation_seconds"))
    replica_stages = re.findall(r"(?m)^STAGE: seed (\d+) \| (train|evaluate) (control|adversarial)", log)
    if replica_stages:
        seed, phase, arm = replica_stages[-1]
        phase_label = "Training" if phase == "train" else "Evaluation"
        arm_label = "fixed" if arm == "control" else "adversarial"
        stamp += f" · Training seed {seed} · {phase_label}, {arm_label} arm"
    status_class = "running" if running else "finished" if status == "Completed" else "idle"
    head = f'<div class="run-top"><div><div class="eyebrow">{"RUNNING" if running else "LATEST RUN"}</div><h2>{esc(task)}</h2><div class="run-date">{esc(stamp)}{" · " + esc(duration) if duration else ""}</div></div><span class="status {status_class}"><i></i>{esc(status)}</span></div>'
    command = metadata.get("command", [])
    script_names = {Path(part).name for part in command}
    if script_names & {"build_adversarial_pool.py", "sample_post_warmup_pool.py"} or "generated_candidates" in data:
        generated = data.get("generated_candidates", 0)
        total = data.get("planned_candidates", 96)
        accepted = data.get("accepted_candidates", 0)
        candidates = rows(target / "candidates.jsonl") if target else []
        accepted_problems = len({row["problem_id"] for row in candidates if row.get("verdict", {}).get("accepted")})
        values = [(f"{generated}<span> / {total}</span>", "Reviewed"), (accepted, "New faulty programs"), (accepted_problems, "Problems with new faults")]
        progress = min(100, 100 * generated / total) if total else 0
        counts = data.get("rejection_and_acceptance_counts", {})
        labels = {"incorrect_but_passes_current_policy_tests": "Accepted", "correct_on_finite_domain": "Correct solution", "restricted_python_rejected": "Unsupported", "caught_by_policy_or_no_valid_policy_tests": "Failed test acceptance criteria", "behavior_duplicate": "Duplicate behavior", "truncated_generation": "Truncated generation"}
        bars = ""
        for reason, label in labels.items():
            count = counts.get(reason, 0)
            if not count:
                continue
            width = 100 * count / max(1, generated)
            tone = REASONS[reason][2]
            bars += f'<div class="distribution-row"><span>{label}</span><div class="distribution-track"><b class="tone-{tone}" style="width:{width:.3f}%"></b></div><strong>{count}</strong></div>'
        body = '<div class="section-label">Candidate outcomes</div>' + (bars or '<div class="empty-copy">Waiting for the first candidate.</div>')
        recent = ""
        for index, row in reversed(list(enumerate(candidates, 1))[-6:]):
            verdict = row.get("verdict", {})
            state, reason, tone = REASONS.get(verdict.get("reason"), ("—", verdict.get("reason", ""), "correct"))
            recent += f'<tr><td class="sample-number">{index:02d}</td><td>{esc(PROBLEMS.get(row.get("problem_id"), row.get("problem_id", "")))}</td><td><span class="candidate-state {"keep" if verdict.get("accepted") else "skip"}">{state}</span><small>{esc(reason)}</small></td></tr>'
        if recent:
            body += '<div class="section-label recent-title">Recent candidates</div><table class="recent-table"><thead><tr><th>Sample</th><th>Problem</th><th>Results</th></tr></thead><tbody>' + recent + '</tbody></table>'
    else:
        stage_log = re.split(r"(?m)^STAGE[^\n]*", log)[-1]
        groups = re.findall(r"(?m)^GROUP (\d+)", stage_log)
        cases = re.findall(r"(?m)^CASE (\d+)/(\d+)", stage_log)
        steps = re.findall(r"steps:\s*(\d+)", stage_log)
        demos = re.findall(r"(?m)^MODEL COMPLETED:", log)
        if metadata.get("kind") == "live_demo":
            completed_files = len(list(target.glob("*/prediction.json"))) if target else 0
            current, total = max(len(demos), completed_files), 3
            values = [(f"{current}<span> / 3</span>", "Models evaluated"), (current * 4, "Generated tests"), ("GPU", "Live inference")]
        elif cases and (not groups or stage_log.rfind("CASE ") > stage_log.rfind("GROUP ")):
            current, total = map(int, cases[-1])
            metrics = data.get("metrics", {})
            values = [(f"{current}<span> / {total}</span>", "Cases in this evaluation"), (metrics.get("valid_tests", "—"), "Valid tests"), (metrics.get("bug_detecting_tests", "—"), "Bug-detecting tests")]
        elif groups:
            current, total = int(groups[-1]), int(steps[-1]) if steps else data.get("optimizer_steps", int(groups[-1]))
            values = [(f"{current}<span> / {total}</span>", "Training groups"), (data.get("groups_with_policy_signal", "—"), "Groups with learning signal"), (data.get("nonzero_gradient_steps", "—"), "Update step")]
            if target and running:
                rollouts = rows(target / "rollouts.jsonl")
                gradients = rows(target / "gradient-audit.jsonl")
                values[1] = (sum(bool(row.get("policy_signal")) for row in rollouts), "Groups with learning signal")
                values[2] = (sum(row.get("gradient_norm", 0) > 0 for row in gradients), "Update step")
        elif cases:
            current, total = map(int, cases[-1])
            metrics = data.get("metrics", {})
            values = [(f"{current}<span> / {total}</span>", "Cases in this evaluation"), (metrics.get("valid_tests", "—"), "Valid tests"), (metrics.get("bug_detecting_tests", "—"), "Bug-detecting tests")]
        else:
            current, total = 0, 1
            values = [("—", "Progress"), ("—", "Results"), ("—", "Time")]
        progress = min(100, 100 * current / max(1, total))
        body = '<div class="empty-copy">' + ("Processing…" if running else "Select a task to start an experiment." if folder is None else "See Results for summary metrics.") + '</div>'
    metrics_html = '<div class="run-metrics">' + ''.join(f'<div><strong>{value}</strong><span class="metric-label">{label}</span></div>' for value, label in values) + '</div>'
    progress_html = f'<div class="progress-track" role="progressbar" aria-valuenow="{progress:.0f}" aria-valuemin="0" aria-valuemax="100"><div style="width:{progress:.2f}%"></div></div>'
    return '<section class="workbench">' + head + metrics_html + progress_html + body + '</section>', log[-24000:], running


def comparison(evidence, split):
    folders = sorted((p for p in evidence.iterdir() if p.is_dir() and (p / "result.json").exists()), reverse=True)
    chosen = None
    for folder in folders:
        data = read_json(folder / "result.json", {})
        value = data.get("comparison")
        if not isinstance(value, dict) or not value.get("protocol_matches") or data.get("preflight_only"):
            continue
        kind = "Train" if folder.name.startswith("train-diagnostic-") else "Validation" if data.get("split") == "validation" else None
        if kind == split:
            chosen = value
            break
    if chosen is None:
        return '<div class="empty-copy">No comparison is available for this split.</div>'
    before, after = chosen["baseline"], chosen["after_static_grpo"]
    definitions = [("Valid tests", "valid_tests", "generated_tests"), ("Correct expected outputs", "correct_expected", "generated_tests"),
                   ("Bug-detecting tests", "bug_detecting_tests", "generated_tests"), ("Faulty variants detected", "cases_detected_with_up_to_4_tests", "evaluated_cases")]
    lines = ""
    for label, numerator, denominator in definitions:
        a, b = 100 * before[numerator] / before[denominator], 100 * after[numerator] / after[denominator]
        difference = b - a
        delta = f"{difference:+.2f}" + " pp" if abs(difference) > 0.005 else "Unchanged"
        tone = "gain" if difference > 0.005 else "loss" if difference < -0.005 else "unchanged"
        lines += f'<tr><th>{label}</th><td><strong>{a:.2f}%</strong><small>{before[numerator]}/{before[denominator]}</small><div class="mini-track"><b style="width:{a:.3f}%"></b></div></td><td><strong>{b:.2f}%</strong><small>{after[numerator]}/{after[denominator]}</small><div class="mini-track after"><b style="width:{b:.3f}%"></b></div></td><td class="{tone}">{delta}</td></tr>'
    note = " · data used during training" if split == "Train" else ""
    return f'<section class="comparison"><div class="comparison-heading"><h2>Before and after GRPO</h2><span>{split}{note}</span></div><table><thead><tr><th>Metric</th><th>Base model</th><th>After GRPO</th><th>Change</th></tr></thead><tbody>{lines}</tbody></table><div class="comparison-meta">{after["generated_tests"]} test · {after["evaluated_cases"]} faulty variants · seed {chosen.get("seed", 42)}</div></section>'
