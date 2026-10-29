import html
import json
import os
import subprocess
import threading
from pathlib import Path

from apps_core import ASSETS, DATA, canonical, digest, domain_text, load_cases, read


PROJECT = Path(__file__).resolve().parents[1]
STATE = PROJECT / "APPS_STATE.json"
TRAIN_PYTHON = ASSETS / "env/Scripts/python.exe"
LOCK = threading.Lock()
TITLE = "Adversarial Reinforcement Learning for Test Case Generation"
CSS = """
.gradio-container { max-width: 1240px !important; margin: auto; background: #fafaf7; }
.apps-header { padding: 20px 0 14px; border-bottom: 1px solid #deded6; margin-bottom: 14px; }
.apps-header h1 { font: 600 26px 'Segoe UI',sans-serif; color: #29372f; margin: 0 0 6px; }
.apps-header p { color: #6c746d; margin: 0; font-size: 14px; }
.apps-case { background: white; border: 1px solid #deded6; padding: 20px; border-radius: 5px; }
.apps-case h2 { font-size: 19px; margin: 0 0 8px; color: #29372f; }
.apps-case details { margin-top: 14px; }
.apps-case pre { white-space: pre-wrap; font: 13px/1.6 Consolas,monospace; background: #f5f6f1; padding: 14px; color: #303a32; }
.apps-compare { display: grid; grid-template-columns: repeat(auto-fit,minmax(290px,1fr)); gap: 16px; }
.apps-result { background: white; border: 1px solid #deded6; border-radius: 5px; padding: 18px; min-width: 0; }
.apps-result h3 { font-size: 17px; color: #29372f; margin: 0 0 14px; }
.apps-sample { border-top: 1px solid #e4e6dd; padding: 12px 0; }
.apps-sample pre { white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.6 Consolas,monospace; background: #f5f6f1; padding: 10px; }
.apps-hit { color: #31674a; font-weight: 600; }
.apps-miss { color: #8b5b35; }
.apps-neutral { color: #58635b; }
.apps-wrong { color: #a04838; }
.apps-muted { font-size: 13px; color: #747c72; }
.apps-value { white-space: pre-wrap; font-family: Consolas,monospace; }
#apps-job-log textarea { height: 300px !important; min-height: 300px !important; max-height: 300px !important; overflow-y: scroll !important; scrollbar-gutter: stable; scrollbar-width: auto; scrollbar-color: #7c897f #e6ebe7; font-family: Consolas,monospace; }
#apps-job-log textarea::-webkit-scrollbar { width: 14px; }
#apps-job-log textarea::-webkit-scrollbar-track { background: #e6ebe7; }
#apps-job-log textarea::-webkit-scrollbar-thumb { background: #7c897f; border: 3px solid #e6ebe7; border-radius: 8px; }
@media(max-width:800px) { .apps-compare { grid-template-columns: 1fr; } }
"""


def state():
    return read(STATE) if STATE.exists() else {}


def validation_cases():
    return load_cases("validation")[0]


def context(identity):
    case = next(case for case in validation_cases() if case["case_id"] == identity)
    return ('<article class="apps-case"><h2>' + html.escape(case["title"]) + '</h2><p class="apps-muted">APPS Train / '
            + str(case["source_id"]) + ' · tập Validation · <a href="' + html.escape(case["source_url"], quote=True)
            + '" target="_blank" rel="noopener">Đề gốc</a></p><h4>Code cần kiểm tra</h4><pre>'
            + html.escape(case["buggy_code"]) + '</pre><details open><summary>Miền input của thí nghiệm</summary><pre>'
            + html.escape(domain_text(case)) + '</pre></details><details><summary>Đề bài APPS nguyên văn</summary><pre>'
            + html.escape(case["specification"]) + '</pre></details></article>')


def predictions(folder):
    path = Path(folder) / "predictions.jsonl"
    return {row["case_id"]: row for row in [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]} if path.exists() else {}


def card(title, row):
    output = '<section class="apps-result"><h3>' + html.escape(title) + '</h3>'
    if not row:
        return output + '<p class="apps-muted">Chưa có kết quả cho bài này.</p></section>'
    for index, sample in enumerate(row["samples"], 1):
        label = "Phát hiện lỗi" if sample["bug_detected"] else "Expected sai" if sample["valid_format"] and not sample["correct_expected"] else "Hợp lệ, chưa phát hiện lỗi" if sample["correct_expected"] else "Test không hợp lệ"
        tone = "apps-hit" if sample["bug_detected"] else "apps-neutral" if sample["correct_expected"] else "apps-wrong" if sample["valid_format"] else "apps-miss"
        output += '<div class="apps-sample"><span class="' + tone + '">Test ' + str(index) + ' · ' + label + '</span><pre>' + html.escape(sample["raw_output"]) + '</pre>'
        if sample.get("valid_format"):
            observed = sample.get("buggy_error") or canonical(sample.get("buggy_output"))
            output += '<div class="apps-muted">Đáp án tham chiếu: <code class="apps-value">' + html.escape(canonical(sample.get("gold_output"))) + '</code><br>Code trả về: <code class="apps-value">' + html.escape(observed) + '</code></div>'
        else:
            reason = "Input sai cấu trúc hoặc nằm ngoài miền cho phép." if sample.get("error") == "schema_or_domain" else "Đầu ra bị cắt trước khi hoàn tất." if sample.get("error") == "truncated" else "Không đọc được một đối tượng JSON đúng định dạng."
            output += '<div class="apps-muted">' + reason + '</div>'
        output += '</div>'
    return output + '</section>'


def saved(identity):
    current = state()
    cards = []
    variants = [("baseline_validation", "Qwen gốc")]
    if current.get("expected_grpo_validation") and current.get("sft_budget_validation"):
        variants += [("sft_budget_validation", "SFT256"), ("expected_grpo_validation", "SFT256 + GRPO96")]
    elif current.get("demo_selection"):
        chosen = current["demo_selection"]
        if chosen["validation_key"] != "baseline_validation":
            variants.append((chosen["validation_key"], chosen["label"]))
        if chosen["validation_key"] != "adversarial_validation" and current.get("adversarial_validation"):
            variants.append(("adversarial_validation", "GRPO đối kháng · 2 vòng"))
    elif current.get("adversarial_validation"):
        variants += [("paired_control_validation", "GRPO cố định · 2 vòng"),
                     ("adversarial_validation", "GRPO đối kháng · 2 vòng")]
    elif current.get("sft_validation"):
        variants.append(("sft_validation", "Sau SFT"))
        variants.append(("pilot_validation", "Sau SFT + GRPO"))
    else:
        variants.append(("pilot_validation", "Sau GRPO pilot"))
    for key, title in variants:
        record = current.get(key)
        rows = predictions(record["path"]) if record else {}
        cards.append(card(title, rows.get(identity)))
    return '<div class="apps-compare">' + ''.join(cards) + '</div>'


def metric_table():
    rows = []
    current = state()
    variants = [("baseline_validation", "Qwen gốc"), ("sft_validation", "Sau SFT"),
                ("sft_budget_validation", "SFT 256 bước · kiểm chứng"),
                ("pilot_validation", "GRPO cố định · 96 bước liền" if current.get("adversarial_validation") else "Sau SFT + GRPO" if current.get("sft_training") else "Sau GRPO pilot"),
                ("paired_control_validation", "GRPO cố định · 2 vòng"),
                ("adversarial_validation", "GRPO đối kháng · 2 vòng"),
                ("expected_grpo_validation", "GRPO từ SFT256 · 96 bước")]
    for key, title in variants:
        record = state().get(key)
        if not record:
            continue
        result = read(Path(record["path"]) / "result.json")
        if digest(Path(record["path"]) / "result.json") != record["result_sha256"]:
            raise ValueError("Saved result hash changed")
        values = result["metrics"]
        rows.append([title, values["generated_tests"], round(values["valid_rate"] * 100, 2),
                     round(values["io_accuracy"] * 100, 2), round(values["attack_rate"] * 100, 2),
                     f'{values["cases_detected"]}/{values["evaluated_cases"]}'])
    return rows


def gpu_busy(ignore_pid=None):
    import psutil
    ignored = {ignore_pid}
    if ignore_pid is not None:
        try:
            ignored.update(process.pid for process in psutil.Process(ignore_pid).parents())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if process.pid in ignored:
                continue
            if (process.info["name"] or "").lower().startswith("python") and any(
                    Path(str(value)).name in ("apps_model_runner.py", "apps_sft.py", "apps_round.py", "apps_adversary.py", "apps_adversarial_round.py", "apps_reasoning_probe.py", "apps_prompt_trial.py", "apps_sft_budget_trial.py", "apps_sft_paired_control.py", "apps_reward_trial.py", "apps_reward_seed_probe.py", "apps_adversarial_review.py", "apps_lora_trial.py", "apps_order_trial.py", "apps_final_evaluate.py", "apps_expected_trial.py", "apps_shared_budget_trial.py", "apps_sft256_rl_followup.py") for value in process.info["cmdline"] or []):
                return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return False


def reward_metric_table():
    path = PROJECT / "results/apps-reward-ablation/metrics.json"
    if not path.exists():
        return []
    report = read(path)
    if digest(Path(report["trial_folder"]) / "result.json") != report["trial_result_sha256"]:
        raise ValueError("Saved reward experiment hash changed")
    labels = {"legacy-v1": "Reward hiện tại · lượt đối chứng mới", "equal-v1": "Reward chia đều"}
    rows = []
    for profile, item in report["runs"].items():
        m = item["validation_metrics"]
        rows.append([labels[profile], round(m["valid_rate"] * 100, 2), round(m["io_accuracy"] * 100, 2),
                     round(m["attack_rate"] * 100, 2), f'{m["cases_detected"]}/{m["evaluated_cases"]}'])
    return rows


def target_order_table():
    path = PROJECT / "results/apps-target-order/metrics.json"
    if not path.exists():
        return []
    report = read(path)
    if digest(Path(report["root"]) / "result.json") != report["trial_result_sha256"]:
        raise ValueError("Saved target-order experiment changed")
    rows = []
    for order, label in (("sorted-v1", "Expected trước"), ("input-first-v1", "Input trước")):
        for stage in ("sft", "grpo"):
            m = report["summary"][order][stage]
            rows.append([label + " · " + stage.upper(), *(f'{m[field]["mean"]:.2f} ± {m[field]["sample_sd"]:.2f}'
                         for field in ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected"))])
    return rows


def input_first_review_table():
    path = PROJECT / "results/apps-input-first-review/metrics.json"
    if not path.exists():
        return []
    report = read(path)
    if digest(Path(report["folder"]) / "result.json") != report["result_sha256"]:
        raise ValueError("Input-first review evidence changed")
    labels = {"base": "Qwen gốc", "sft": "SFT64 input-first", "control": "GRPO cố định · 2 vòng", "adaptive": "GRPO đối kháng · 2 vòng"}
    return [[labels[name]] + [f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}' for f in
                             ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")]
            for name, values in report["summary"].items()]


def shared_budget_table():
    path = PROJECT / "results/apps-shared-sft-budget/metrics.json"
    if not path.exists():
        return []
    report = read(path)
    if digest(Path(report["root"]) / "result.json") != report["trial_result_sha256"]:
        raise ValueError("Shared SFT budget evidence changed")
    return [["SFT " + steps + " bước"] + [f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}' for f in
                                         ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")]
            for steps, values in report["summary"].items()]


def sft256_rl_table():
    path = PROJECT / "results/apps-sft256-grpo/metrics.json"
    if not path.exists():
        return []
    report = read(path)
    if digest(Path(report["root"]) / "result.json") != report["result_sha256"]:
        raise ValueError("SFT256 RL evidence changed")
    return [["SFT256" if name == "sft256" else "SFT256 + GRPO96"] + [f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}' for f in
                                                                   ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")]
            for name, values in report["summary"].items()]


def training_plot(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from export_tensorboard import scalar_rows
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor="#fafaf7")
    rows = scalar_rows(Path(folder))
    for axis, tag, title, xlabel, ylabel in (
        (axes[0], "rollout/reward_mean", "Reward trong lúc học", "Nhóm sinh test", "Reward trung bình (0–1)"),
        (axes[1], "gradient/norm_observed", "Gradient của adapter", "Bước cập nhật", "Độ lớn gradient"),
    ):
        points = [(step, value) for key, step, value in rows if key == tag]
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], color="#4c705b", linewidth=1.5)
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def training_view():
    current = state()
    candidates = [(key, current[key]) for key in ("adversarial_training", "pilot_training", "expected_grpo_training") if current.get(key)]
    if not candidates:
        return "Chưa hoàn tất lượt GRPO trên APPS.", None, {}
    key, record = max(candidates, key=lambda item: read(Path(item[1]["path"]) / "result.json").get("started_at", ""))
    result = read(Path(record["path"]) / "result.json")
    text = f'**GRPO:** {result["optimizer_steps"]} bước · {result["nonzero_gradient_steps"]} bước có gradient khác 0 · {len(result["changed_adapter_tensors"])} tensor adapter thay đổi.'
    if key == "expected_grpo_training":
        text += "\n\nKhởi tạo từ SFT256 input-first, code Train cố định. Đây là phép đo phần thêm của RL, không phải vòng đối kháng mới."
    elif key == "adversarial_training":
        experiment = read(Path(current["adversarial_round_path"]) / "result.json")
        text += f'\n\nĐây là vòng cuối của {len(experiment["rounds"])} vòng đối kháng; tổng {experiment["total_steps_per_arm"]} bước GRPO. Biểu đồ bên dưới thể hiện vòng cuối.'
        text += "\n\n| Vòng | Code sinh | Qua bộ lọc | Vị trí Train đối kháng /41 |\n|---|---:|---:|---:|"
        for stage in experiment["rounds"]:
            pool = read(Path(stage["pool"]) / "result.json")
            text += f'\n| {stage["round"]} | {pool["sampled_candidates"]} | {pool["accepted_candidates"]} | {stage["curricula"]["adaptive"]["counts"].get("local_adversary", 0)} |'
    elif record and current.get("sft_training"):
        text += "\n\nKhởi tạo từ adapter SFT trên APPS Train; optimizer GRPO mới."
    return text, training_plot(record["path"]), result


def run_live(identity, variant):
    if not LOCK.acquire(blocking=False):
        yield "Một lượt chạy đang hoạt động.", ""
        return
    try:
        if gpu_busy():
            yield "GPU đang chạy một thí nghiệm. Hãy đợi tác vụ đó hoàn tất.", ""
            return
        command = [str(TRAIN_PYTHON), "-B", str(PROJECT / "scripts/apps_model_runner.py"), "evaluate", "--case", identity]
        if variant != "Qwen gốc":
            key = {"Sau SFT": "sft_training", "GRPO cố định · 2 vòng": "paired_control_training",
                   "GRPO đối kháng · 2 vòng": "adversarial_training", "SFT 256 bước · kiểm chứng": "sft_budget_training",
                   "GRPO từ SFT256": "expected_grpo_training"}.get(variant, "pilot_training")
            pilot = state().get(key)
            if not pilot:
                yield "Chưa có adapter APPS.", ""
                return
            adapter = Path(pilot["adapter_path"])
            if digest(adapter / "adapter_model.safetensors") != pilot["adapter_sha256"]:
                raise RuntimeError("Adapter hash changed")
            command += ["--adapter", str(adapter)]
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
        process = subprocess.Popen(command, cwd=str(PROJECT), env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        folder = None
        log = ""
        for line in process.stdout:
            log = (log + line)[-18000:]
            if line.startswith("EVIDENCE: "):
                folder = Path(line.split("EVIDENCE: ", 1)[1].strip())
                if not folder.resolve().is_relative_to((ASSETS / "evidence").resolve()):
                    raise RuntimeError("Unexpected output path")
            yield "Đang sinh và kiểm chứng test trên GPU…", log
        if process.wait() != 0 or folder is None:
            yield "Lượt chạy chưa hoàn tất; xem log bên dưới.", log
            return
        result = read(folder / "result.json")
        if result["status"] != "evaluation_completed":
            raise RuntimeError("Incomplete live result")
        yield card(variant + " · chạy trực tiếp", predictions(folder).get(identity)), log
    finally:
        LOCK.release()


def build_ui():
    import gradio as gr
    import apps_jobs
    cases = validation_cases()
    choices = [(case["title"] + " · " + case["case_id"], case["case_id"]) for case in cases]
    identity = next((case["case_id"] for case in cases if case["case_id"] == "apps-train-3014/m1"), cases[0]["case_id"])
    manifest = read(DATA / "manifest.json")
    with gr.Blocks(title=TITLE, analytics_enabled=False) as app:
        gr.HTML('<header class="apps-header"><h1>' + TITLE + '</h1><p>Huấn luyện tăng cường đối kháng để sinh ca kiểm thử · APPS Introductory</p></header>')
        with gr.Tab("Demo"):
            selected = gr.Dropdown(choices, value=identity, label="Bài toán")
            problem = gr.HTML(context(identity))
            gr.Markdown("#### Kết quả đã lưu")
            comparison = gr.HTML(saved(identity))
            with gr.Accordion("Chạy trực tiếp trên máy", open=False):
                variants = ["Qwen gốc", "Sau SFT", "GRPO cố định · 96 bước liền", "GRPO cố định · 2 vòng", "GRPO đối kháng · 2 vòng"] if state().get("adversarial_validation") else ["Qwen gốc", "Sau SFT", "Sau SFT + GRPO"] if state().get("sft_training") else ["Qwen gốc", "Sau GRPO pilot"]
                if state().get("sft_budget_training"):
                    variants.insert(2, "SFT 256 bước · kiểm chứng")
                if state().get("expected_grpo_training"):
                    variants.append("GRPO từ SFT256")
                variant = gr.Radio(variants, value="Qwen gốc", label="Phiên bản")
                button = gr.Button("Sinh 4 test", variant="primary")
                live = gr.HTML()
                with gr.Accordion("Log", open=False):
                    log = gr.Textbox(lines=8, interactive=False, show_label=False)
                button.click(run_live, [selected, variant], [live, log], concurrency_limit=1)
            selected.change(lambda value: (context(value), saved(value), "", ""), selected, [problem, comparison, live, log])
        with gr.Tab("Kết quả"):
            table = gr.Dataframe(value=metric_table(), headers=["Phiên bản", "Test sinh", "Hợp lệ (%)", "Expected đúng (%)", "Test bắt lỗi (%)", "Code phát hiện @4"], interactive=False)
            refresh = gr.Button("Cập nhật kết quả")
            refresh.click(metric_table, outputs=table)
            gr.Markdown("Mỗi code Validation sinh 4 test, seed 42. Mẫu sai và bị cắt vẫn nằm trong mẫu số. Hai nhánh GRPO 2 vòng cùng bắt đầu từ SFT và cùng ngân sách học. Mốc 96 bước liền là lượt thực nghiệm trước đó." if state().get("adversarial_validation") else "Mỗi code Validation sinh 4 test, seed 42. Mẫu sai và bị cắt vẫn nằm trong mẫu số. Các mốc SFT và GRPO được đánh giá riêng. Vòng sinh code đối kháng trên APPS đang được hoàn thiện.")
            if state().get("sft_budget_validation"):
                gr.Markdown("Hai nhánh GRPO theo vòng dùng cùng SFT64 cha. SFT256 và GRPO từ SFT256 là phép thử thời lượng học và phần thêm của RL riêng." if state().get("expected_grpo_validation") else "Mốc SFT 256 bước được chấm riêng để kiểm chứng ngân sách học. Các nhánh GRPO hiện tại vẫn xuất phát từ SFT 64 bước.")
            if (PROJECT / "results/apps-reward-ablation/metrics.json").exists():
                with gr.Accordion("Kiểm chứng tỷ trọng reward", open=False):
                    gr.Markdown("Hai lượt GRPO mới, cùng SFT 64 bước và 96 bước học trên code cố định. Bảng dùng seed sinh 42; đây là phép thử riêng về reward, chưa thay các checkpoint chính.")
                    gr.Dataframe(value=reward_metric_table(), headers=["Nhánh", "Hợp lệ (%)", "Expected đúng (%)", "Test bắt lỗi (%)", "Code phát hiện @4"], interactive=False)
                    picture = PROJECT / "results/apps-reward-ablation/sampling-variability.png"
                    if picture.exists():
                        gr.Markdown("Ba seed sinh test 42/43/44 trên cùng checkpoint; chỉ có một seed huấn luyện. Toàn bộ kết quả được giữ, không chọn seed thuận lợi.")
                        gr.Image(str(picture), show_label=False, interactive=False)
            if (PROJECT / "results/apps-target-order/metrics.json").exists():
                with gr.Accordion("Thứ tự nhãn SFT", open=False):
                    gr.Markdown("Cùng input/expected và ngân sách học, khác thứ tự hai trường trong nhãn SFT. Trung bình ± SD của ba seed sinh test; một seed huấn luyện.")
                    gr.Dataframe(value=target_order_table(), headers=["Nhánh", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code phát hiện /15"], interactive=False)
                    gr.Image(str(PROJECT / "results/apps-target-order/comparison.png"), show_label=False, interactive=False)
            if (PROJECT / "results/apps-input-first-review/metrics.json").exists():
                with gr.Accordion("Đối kháng từ nhãn input-first", open=False):
                    gr.Markdown("Cùng SFT64 cha, hai nhánh cùng 2 × 48 bước GRPO. Trung bình ± SD qua ba seed sinh; một seed train.")
                    gr.Dataframe(value=input_first_review_table(), headers=["Checkpoint", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code phát hiện /15"], interactive=False)
                    gr.Image(str(PROJECT / "results/apps-input-first-review/sampling.png"), show_label=False, interactive=False)
            if (PROJECT / "results/apps-expected-diagnosis/analysis.json").exists():
                with gr.Accordion("Expected sai ở đâu?", open=False):
                    gr.Image(str(PROJECT / "results/apps-expected-diagnosis/expected-errors.png"), show_label=False, interactive=False)
                    gr.Markdown("GRPO96 input-first: 36/180 input làm lộ lỗi nhưng expected sai. Đáp án tham chiếu chỉ dùng để chấm; đầu ra model được giữ nguyên.")
            if (PROJECT / "results/apps-shared-sft-budget/metrics.json").exists():
                with gr.Accordion("Thời lượng học expected", open=False):
                    gr.Markdown("Checkpoint 64 và 256 từ cùng một lượt SFT. Bảng Validation là trung bình ± SD qua ba seed sinh. Đây là kiểm chứng SFT, chưa phải RL.")
                    gr.Dataframe(value=shared_budget_table(), headers=["Checkpoint", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code phát hiện /15"], interactive=False)
                    gr.Image(str(PROJECT / "results/apps-shared-sft-budget/learning-curve.png"), show_label=False, interactive=False)
            if (PROJECT / "results/apps-sft256-grpo/metrics.json").exists():
                with gr.Accordion("GRPO thêm gì sau SFT256?", open=False):
                    gr.Markdown("Cùng parent SFT256, thêm 96 bước GRPO trên code Train cố định. Ba seed sinh, một seed train; chưa phải thực nghiệm đối kháng mới.")
                    gr.Dataframe(value=sft256_rl_table(), headers=["Checkpoint", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code phát hiện /15"], interactive=False)
                    gr.Image(str(PROJECT / "results/apps-sft256-grpo/comparison.png"), show_label=False, interactive=False)
        with gr.Tab("Dữ liệu"):
            gr.Dataframe(value=[[name, spec["problems"], spec["cases"]] for name, spec in manifest["splits"].items()], headers=["Tập phát triển", "Bài APPS", "Code lỗi"], interactive=False)
            gr.Markdown("Nguồn: **codeparrot/apps**, nhãn Introductory. Tập con dùng bài dạng gọi hàm và miền input hữu hạn được ghi rõ. Code lỗi được tạo bằng thay đổi AST từ lời giải nguồn, không phải code lỗi có sẵn của APPS.\n\nTrain/Validation/Test phát triển được chia theo bài trong **APPS Train**. Tập Test chính thức của APPS chưa tải, chưa đánh giá. Test phát triển chưa được model đánh giá.\n\nHai lời giải nguồn phải qua test gốc và đồng ý trên toàn bộ miền input thực nghiệm. Bài không đạt bộ lọc được ghi lý do loại.")
            gr.JSON(manifest, label="Nguồn và phạm vi")
        with gr.Tab("Thực nghiệm"):
            task = gr.Dropdown(apps_jobs.TASKS, value=apps_jobs.TASKS[0], label="Tác vụ")
            steps = gr.Slider(minimum=1, maximum=96, step=1, value=48, label="Số bước GRPO", visible=False)
            note = gr.Markdown("Kiểm tra nguồn dữ liệu, tập chia và bộ chấm; không tải model.")
            def task_options(value):
                notes = {
                    apps_jobs.TASKS[0]: "Kiểm tra nguồn dữ liệu, tập chia và bộ chấm; không tải model.",
                    apps_jobs.TASKS[1]: "Sinh và chấm 4 test/code trên Validation bằng Qwen gốc. Không huấn luyện.",
                    apps_jobs.TASKS[2]: "Học trên Train từ adapter SFT đã lưu, với optimizer GRPO mới. Sau khi học, chọn Chấm adapter hiện tại trên Validation để đo chất lượng." if state().get("sft_training") else "Tạo LoRA mới từ Qwen gốc và học trên Train; không học tiếp adapter cũ. Sau đó chọn chấm adapter để đo chất lượng.",
                    apps_jobs.TASKS[3]: "Chấm adapter của lượt GRPO mới nhất trên cùng Validation, seed 42. Không cập nhật trọng số.",
                    apps_jobs.TASKS[4]: "Chạy từ Qwen gốc: chấm baseline, SFT 64 bước trên Train, rồi GRPO theo số bước đã chọn. Chấm Validation tại từng mốc; không dùng Test.",
                }
                return gr.Slider(visible=value in (apps_jobs.TASKS[2], apps_jobs.TASKS[4])), notes[value]
            task.change(task_options, task, [steps, note])
            with gr.Row():
                start = gr.Button("Bắt đầu", variant="primary")
                stop = gr.Button("Dừng tác vụ")
            status = gr.Textbox(label="Trạng thái", value=apps_jobs.snapshot()[0], interactive=False)
            gr.Markdown("Dừng sẽ mất phần cập nhật chưa lưu của lượt đang chạy; các kết quả đã lưu vẫn được giữ.")
            with gr.Accordion("Log và nơi lưu kết quả", open=False):
                automatic = gr.Checkbox(value=True, label="Tự cập nhật log")
                console = gr.Textbox(label="Log", value=apps_jobs.snapshot()[1], lines=12, max_lines=12,
                                     interactive=False, autoscroll=False, elem_id="apps-job-log")
                with gr.Row():
                    first_line = gr.Button("Đầu log")
                    last_line = gr.Button("Cuối log")
                    update_log = gr.Button("Cập nhật log")
                first_line.click(fn=None, js="() => { const log = document.querySelector('#apps-job-log textarea'); if (log) log.scrollTop = 0; }")
                last_line.click(fn=None, js="() => { const log = document.querySelector('#apps-job-log textarea'); if (log) log.scrollTop = log.scrollHeight; }")
                folder = gr.Textbox(label="Thư mục lượt chạy", value=apps_jobs.snapshot()[2], interactive=False)
            outputs = [status, console, folder]
            start.click(lambda name, count: apps_jobs.start(name, count, LOCK, gpu_busy), [task, steps], outputs, concurrency_limit=1)
            stop.click(apps_jobs.stop, outputs=outputs, queue=False)
            def poll_job(enabled):
                current_status, current_log, current_folder = apps_jobs.snapshot()
                return current_status, current_log if enabled else gr.skip(), current_folder
            update_log.click(apps_jobs.snapshot, outputs=outputs, queue=False)
            gr.Timer(2).tick(poll_job, inputs=automatic, outputs=outputs, show_progress="hidden", queue=False)
        with gr.Tab("Huấn luyện"):
            initial_text, initial_plot, initial_result = training_view()
            summary = gr.Markdown(initial_text)
            plot = gr.Plot(initial_plot, show_label=False)
            refresh_training = gr.Button("Cập nhật lượt mới")
            with gr.Accordion("Thông số lượt học", open=False):
                training_data = gr.JSON(initial_result, show_label=False)
            refresh_training.click(training_view, outputs=[summary, plot, training_data])
            gr.Markdown("[Mở TensorBoard](http://127.0.0.1:6006)")
    return app
