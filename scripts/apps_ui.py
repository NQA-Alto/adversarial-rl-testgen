import json
import os
import subprocess
import threading
from pathlib import Path

from apps_core import ASSETS, DATA, digest, load_cases, read
from apps_presentation import comparison, problem, result_card, selected_key, study, summary_table, variants


PROJECT = Path(__file__).resolve().parents[1]
STATE = PROJECT / "APPS_STATE.json"
TRAIN_PYTHON = ASSETS / "env/Scripts/python.exe"
LOCK = threading.Lock()
TITLE = "Huấn luyện tăng cường đối kháng để sinh ca kiểm thử"
CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; background: #fafaf7; --block-label-text-color: #59665c; --button-primary-background-fill: #365b46; --button-primary-background-fill-hover: #284b37; --button-primary-border-color: #365b46; }
.gradio-container button { box-shadow: none !important; }
.apps-header { padding: 22px 0 18px; border-bottom: 1px solid #dcded5; margin-bottom: 14px; }
.apps-header h1 { font: 600 25px/1.4 'Segoe UI',sans-serif; color: #29372f; margin: 0 0 5px; }
.apps-header p { color: #697368; margin: 0; font-size: 13px; }
.apps-case { background: #fff; border: 1px solid #dcded5; padding: 22px; border-radius: 4px; }
.apps-case h2 { font-size: 20px; margin: 0 0 5px; color: #29372f; }
.apps-case h4 { font-size: 13px; font-weight: 600; margin: 16px 0 8px; color: #5e695e; }
.apps-case pre { white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.7 Consolas,monospace; background: #f4f5f0; padding: 14px; margin: 0; color: #303a32; }
.apps-problem-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; }
.apps-case details { margin-top: 16px; }
.apps-case details pre { margin-top: 10px; max-height: 350px; overflow-y: auto; }
.apps-muted { font-size: 13px; color: #687164; margin: 0; }
.apps-muted a { color: #365b46; }
.apps-compare { display: grid; grid-template-columns: repeat(auto-fit,minmax(360px,1fr)); gap: 16px; }
.apps-result { background: #fff; border: 1px solid #dcded5; border-radius: 4px; padding: 18px; min-width: 0; }
.apps-result h3 { font-size: 17px; font-weight: 600; color: #29372f; margin: 0 0 6px; }
.apps-result-count { color: #365b46; font-size: 13px; margin: 0 0 15px; }
.apps-result-count span { color: #687164; }
.apps-table-wrap { overflow-x: auto; }
.apps-summary-table { border-collapse: collapse; width: 100%; font-size: 14px; background: #fff; }
.apps-summary-table th { padding: 13px 16px; text-align: right; font-size: 12px; color: #687164; font-weight: 500; border-bottom: 1px solid #dcded5; }
.apps-summary-table td { padding: 14px 16px; text-align: right; border-bottom: 1px solid #e6e8df; color: #303a32; }
.apps-summary-table td:first-child,.apps-summary-table th:first-child { text-align: left; }
.apps-summary-table strong { font-weight: 650; color: #365b46; }
.apps-selected-row { background: #f0f4ec; }
.apps-test-table { border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 12px; }
.apps-test-table th { text-align: left; padding: 9px 6px; color: #687164; font-weight: 500; background: #f5f6f1; }
.apps-test-table th:first-child { width: 49%; }
.apps-test-table td { vertical-align: top; padding: 12px 6px; border-bottom: 1px solid #e7e9e0; }
.apps-test-table code { font: 12px/1.6 Consolas,monospace; white-space: pre-wrap; overflow-wrap: anywhere; color: #303a32; background: none; }
.apps-test-table small { display: block; font-size: 11px; margin-top: 5px; color: #a04838; }
.apps-test-index { color: #7c8378; display: block; margin-bottom: 4px; }
.apps-test-status { display: block; font-size: 11px; margin-top: 8px; }
.apps-hit { color: #2f6446; font-weight: 600; }
.apps-neutral { color: #687164; }
.apps-wrong { color: #a04838; }
.apps-raw { color: #687164; font-size: 12px; margin-top: 15px; }
.apps-raw pre { font: 12px/1.6 Consolas,monospace; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 280px; overflow-y: auto; color: #303a32; background: #f5f6f1; padding: 12px; }
#apps-job-log textarea { height: 280px !important; min-height: 280px !important; max-height: 280px !important; overflow-y: scroll !important; scrollbar-gutter: stable; scrollbar-width: auto; scrollbar-color: #7c897f #e6ebe7; font-family: Consolas,monospace; }
#apps-job-log textarea::-webkit-scrollbar { width: 14px; }
#apps-job-log textarea::-webkit-scrollbar-thumb { background: #7c897f; border: 3px solid #e6ebe7; border-radius: 7px; }
@media(max-width:800px) { .apps-compare,.apps-problem-grid { grid-template-columns: 1fr; } .apps-header h1 { font-size: 21px; } }
"""


def state():
    return read(STATE) if STATE.exists() else {}


def validation_cases():
    return load_cases("validation")[0]


def context(identity):
    return problem(next(case for case in validation_cases() if case["case_id"] == identity))


def predictions(folder):
    path = Path(folder) / "predictions.jsonl"
    return {row["case_id"]: row for row in [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]} if path.exists() else {}


def card(title, row):
    return result_card(title, row)


def saved(identity, version=None):
    current = state()
    return comparison(current, identity, version or selected_key(current))


def gpu_busy(ignore_pid=None):
    import psutil
    ignored = {ignore_pid}
    if ignore_pid is not None:
        try:
            ignored.update(process.pid for process in psutil.Process(ignore_pid).parents())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    names = {"apps_model_runner.py", "apps_sft.py", "apps_round.py", "apps_adversary.py", "apps_adversarial_round.py", "apps_reasoning_probe.py", "apps_prompt_trial.py", "apps_sft_budget_trial.py", "apps_sft_paired_control.py", "apps_reward_trial.py", "apps_reward_seed_probe.py", "apps_adversarial_review.py", "apps_lora_trial.py", "apps_order_trial.py", "apps_final_evaluate.py", "apps_expected_trial.py", "apps_shared_budget_trial.py", "apps_sft256_rl_followup.py", "apps_lora_input_trial.py", "resume_sft256_evaluation.py", "run_capability.py", "train_expanded_candidate.py", "bounded_expected.py", "expanded_grpo_worker.py", "oracle_generation_worker.py", "finish_expanded_study.py", "run_balanced_study.py", "apps_balanced_study.py"}
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if process.pid not in ignored and (process.info["name"] or "").lower().startswith("python") and any(Path(str(value)).name in names for value in process.info["cmdline"] or []):
                return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return False


def training_plot(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from export_tensorboard import scalar_rows
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor="#fafaf7")
    rows = scalar_rows(Path(folder))
    for axis, tag, title, xlabel, ylabel in (
        (axes[0], "rollout/reward_mean", "Reward", "Nhóm sinh test", "Reward trung bình (0–1)"),
        (axes[1], "gradient/norm_observed", "Cập nhật adapter", "Bước cập nhật", "Độ lớn gradient"),
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
    if current.get("selected_local_model"):
        current = read(Path(current["selected_local_model"]["state_path"]))
    records = [current[key] for key in ("adversarial_training", "pilot_training") if current.get(key)]
    report = study()
    if report:
        records.append(report["last_training"])
    if not records:
        return "Chưa có lượt huấn luyện.", None, {}
    record = max(records, key=lambda item: read(Path(item["path"]) / "result.json").get("started_at", ""))
    result = read(Path(record["path"]) / "result.json")
    text = f'**GRPO · {result["optimizer_steps"]} bước** · {result["nonzero_gradient_steps"]} bước có cập nhật gradient'
    if report and record["path"] == report["last_training"]["path"]:
        text = f'**GRPO đối kháng · vòng 2/2 · {result["optimizer_steps"]} bước** · {result["nonzero_gradient_steps"]} bước có cập nhật gradient'
    return text, training_plot(record["path"]), result


def run_live(identity, variant):
    if not LOCK.acquire(blocking=False):
        yield "Một tác vụ đang chạy.", ""
        return
    try:
        if gpu_busy():
            yield "GPU đang chạy thực nghiệm. Hãy đợi lượt hiện tại hoàn tất.", ""
            return
        current = state()
        if variant == current.get("selected_local_model", {}).get("label"):
            variant = selected_key(current)
        version = next(item for item in variants(current) if item["key"] == variant)
        command = [str(TRAIN_PYTHON), "-B", str(PROJECT / "scripts/apps_model_runner.py"), "evaluate", "--case", identity,
                   "--model-profile", "2b", "--prompt-version", "apps-call-v2", "--no-publish"]
        if version["adapter"]:
            adapter = Path(version["adapter"])
            if version.get("adapter_sha256") and digest(adapter / "adapter_model.safetensors") != version["adapter_sha256"]:
                raise ValueError("Adapter không khớp bản đã lưu.")
            command += ["--adapter", str(adapter)]
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                           TEMP=str(ASSETS / "temp"), TMP=str(ASSETS / "temp"))
        chosen = current.get("selected_local_model")
        if chosen:
            environment.update(TTNT_APPS_DATA_DIR=chosen["dataset_path"], TTNT_APPS_STATE_PATH=chosen["state_path"], TTNT_MODEL_PROFILE="2b")
        process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        folder, log = None, ""
        for line in process.stdout:
            log = (log + line)[-18000:]
            if line.startswith("EVIDENCE: "):
                folder = Path(line.removeprefix("EVIDENCE: ").strip())
                if not folder.resolve().is_relative_to((ASSETS / "evidence").resolve()):
                    raise ValueError("Đường dẫn kết quả không hợp lệ.")
            yield "Đang sinh test…", log
        if process.wait() != 0 or folder is None:
            yield "Lượt chạy chưa hoàn tất. Xem nhật ký bên dưới.", log
            return
        result = read(folder / "result.json")
        if result["status"] != "evaluation_completed":
            raise ValueError("Lượt chạy chưa hoàn tất.")
        yield card(version["label"] + " · chạy trực tiếp", predictions(folder).get(identity)), log
    finally:
        LOCK.release()


def build_ui():
    import gradio as gr
    import apps_jobs
    current, report = state(), study()
    cases = validation_cases()
    choices = [(case["title"] + " · Lỗi " + case["case_id"].rsplit("/m", 1)[-1], case["case_id"]) for case in cases]
    identity = next((case["case_id"] for case in cases if case["case_id"] == "apps-train-2968/m1"), cases[0]["case_id"])
    versions = variants(current)
    selected_model = current.get("selected_local_model")
    manifest = read(Path(selected_model["dataset_path"]) / "manifest.json") if selected_model else read(DATA / "manifest.json")
    with gr.Blocks(title=TITLE, analytics_enabled=False) as app:
        gr.HTML('<header class="apps-header"><h1>' + TITLE + '</h1><p>APPS Introductory · Qwen3.5-2B</p></header>')
        with gr.Tab("Demo"):
            selected = gr.Dropdown(choices, value=identity, label="Bài toán")
            details = gr.HTML(context(identity))
            with gr.Row():
                version = gr.Radio([(v["label"], v["key"]) for v in versions], value=selected_key(current), label="Phiên bản", scale=4)
                button = gr.Button("Sinh 4 test mới", variant="primary", scale=1)
            gr.Markdown("#### Kết quả đã lưu · Validation")
            comparison_view = gr.HTML(saved(identity))
            live = gr.HTML()
            with gr.Accordion("Nhật ký chạy trực tiếp", open=False):
                log = gr.Textbox(lines=8, max_lines=8, interactive=False, show_label=False)
            selected.change(lambda name, key: (context(name), saved(name, key), "", ""), [selected, version], [details, comparison_view, live, log])
            version.change(lambda name, key: (saved(name, key), "", ""), [selected, version], [comparison_view, live, log])
            button.click(run_live, [selected, version], [live, log], concurrency_limit=1)
        with gr.Tab("Kết quả"):
            if report:
                gr.Markdown("### Đánh giá cuối trên Test")
                chosen_label = next(item["label"] for item in versions if item["key"] == report["selected_variant"])
                gr.Markdown("Phiên bản sử dụng: **" + chosen_label + "**, chọn trên Validation.")
                gr.HTML(summary_table(report, "test"))
                gr.Image(report["test_chart_path"], show_label=False, interactive=False)
                gr.Markdown("Trung bình ± SD qua ba seed sinh 42/43/44; một seed huấn luyện. Mỗi code sinh 4 test. Mẫu không hợp lệ vẫn tính trong mẫu số.")
                with gr.Accordion("Validation và thiết lập so sánh", open=False):
                    gr.HTML(summary_table(report, "validation"))
                    gr.Image(report["validation_chart_path"], show_label=False, interactive=False)
                    gr.Markdown("Hai nhánh GRPO dùng cùng SFT840 cha, cùng reward và cùng 2 × 32 bước học. Nhánh đối kháng làm mới code sau vòng đầu. Checkpoint được chọn trên Validation trước khi mở Test.")
                    gr.Dataframe(value=report["round_rows"], headers=["Vòng", "Code ứng viên", "Được nhận", "Vị trí đối kháng /145", "Test tiếp xúc khi học"], interactive=False)
                with gr.Accordion("Phân tích kết quả test", open=False):
                    gr.Dataframe(value=report["test_error_rows"], headers=["Phiên bản", "Phát hiện lỗi", "Đúng, chưa bắt lỗi", "Expected sai", "Không hợp lệ"], interactive=False)
                    gr.Markdown("Số lượng trên 216 test mỗi phiên bản, gộp ba seed sinh.")
            else:
                gr.Markdown("### Kiểm chứng checkpoint")
                if selected_model:
                    values = selected_model["mean"]
                    gr.Dataframe(value=[["SFT840", round(values["correct_expected"], 2), round(values["cases_detected"], 2)]], headers=["Phiên bản", "Expected đúng /60", "Code phát hiện /15"], interactive=False)
                    gr.Image(selected_model["chart_path"], show_label=False, interactive=False)
                gr.Markdown("Vòng đối kháng trên tập mở rộng đang chạy. Kết quả sẽ cập nhật sau khi kiểm chứng hoàn tất.")
        with gr.Tab("Dữ liệu"):
            gr.Dataframe(value=[[name, spec["problems"], spec["cases"]] for name, spec in manifest["splits"].items()], headers=["Tập dữ liệu", "Bài toán", "Code lỗi"], interactive=False)
            gr.Markdown("Nguồn: [APPS — codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps), mức Introductory. Dùng tập con Python dạng gọi hàm, với miền input hữu hạn. Các tập chia theo bài toán trong APPS Train; đây không phải toàn bộ benchmark APPS.")
            with gr.Accordion("Chuẩn bị và chia dữ liệu", open=False):
                gr.Markdown("Code lỗi ban đầu tạo bằng thay đổi AST từ lời giải nguồn. Hai lời giải phải qua test gốc và đồng ý trên miền thực nghiệm. Code đối kháng sinh trong lúc học được lọc riêng. Tập Test chính thức của APPS không được sử dụng.")
                gr.JSON(manifest, show_label=False)
        with gr.Tab("Thực nghiệm"):
            gr.Markdown("### Chạy tác vụ")
            tasks = apps_jobs.TASKS[:4] if selected_model else apps_jobs.TASKS
            task = gr.Dropdown(tasks, value=tasks[0], label="Tác vụ")
            steps = gr.Slider(minimum=1, maximum=96, step=1, value=32, label="Số bước GRPO", visible=False)
            scope = gr.Markdown("GRPO trên code Train cố định, khởi tạo từ adapter SFT đã lưu.", visible=False)
            task.change(lambda value: (gr.update(visible=value == apps_jobs.TASKS[2]), gr.update(visible=value == apps_jobs.TASKS[2])), task, [steps, scope])
            with gr.Row():
                start = gr.Button("Bắt đầu", variant="primary")
                stop = gr.Button("Dừng")
            status = gr.Textbox(value=apps_jobs.snapshot()[0], label="Trạng thái", interactive=False)
            with gr.Accordion("Nhật ký và kết quả", open=False):
                automatic = gr.Checkbox(value=True, label="Tự cập nhật")
                console = gr.Textbox(value=apps_jobs.snapshot()[1], lines=12, max_lines=12, label="Log", interactive=False, autoscroll=False, elem_id="apps-job-log")
                with gr.Row():
                    first = gr.Button("Đầu log")
                    last = gr.Button("Cuối log")
                    refresh = gr.Button("Cập nhật")
                first.click(fn=None, js="() => { const el = document.querySelector('#apps-job-log textarea'); if (el) el.scrollTop = 0; }")
                last.click(fn=None, js="() => { const el = document.querySelector('#apps-job-log textarea'); if (el) el.scrollTop = el.scrollHeight; }")
                folder = gr.Textbox(value=apps_jobs.snapshot()[2], label="Nơi lưu", interactive=False)
                gr.Markdown("Dừng sẽ giữ kết quả đã lưu; cập nhật chưa lưu của lượt hiện tại sẽ mất.")
            outputs = [status, console, folder]
            start.click(lambda name, count: apps_jobs.start(name, count, LOCK, gpu_busy), [task, steps], outputs, concurrency_limit=1)
            stop.click(apps_jobs.stop, outputs=outputs, queue=False)
            refresh.click(apps_jobs.snapshot, outputs=outputs, queue=False)
            def poll(enabled):
                a, b, c = apps_jobs.snapshot()
                return a, b if enabled else gr.skip(), c
            gr.Timer(2).tick(poll, automatic, outputs, show_progress="hidden", queue=False)
            with gr.Accordion("Đường học GRPO", open=False):
                text, figure, info = training_view()
                summary = gr.Markdown(text)
                plot = gr.Plot(figure, show_label=False)
                update = gr.Button("Cập nhật lượt học")
                with gr.Accordion("Thông số", open=False):
                    raw = gr.JSON(info, show_label=False)
                    gr.Markdown("[Mở TensorBoard](http://127.0.0.1:6006)")
                update.click(training_view, outputs=[summary, plot, raw])
    return app
