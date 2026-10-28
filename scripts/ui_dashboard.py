import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


ASSETS = Path(os.environ.get("TTNT_ASSETS", r"D:\DA\TTNT_De19_Assets"))
PROJECT = Path(os.environ.get("TTNT_PROJECT", r"D:\DA\DoAn_TTNT_De19\adversarial-rl-testgen"))
EVIDENCE = ASSETS / "evidence"
TRAIN_PYTHON = ASSETS / "env/Scripts/python.exe"
SCRIPTS = PROJECT / "scripts"
TASKS = {
    "Kiểm tra GRPO với seed huấn luyện 43 và 44": "run_grpo_replicates.py",
    "Sinh 256 ứng viên đối kháng sau SFT": "sample_post_warmup_pool.py",
    "Huấn luyện hai nhánh GRPO — 96 bước mỗi nhánh": "run_post_warmup_round.py",
    "Kiểm tra model local": "check_local_model.py",
}
for key, value in {"GRADIO_ANALYTICS_ENABLED": "False", "GRADIO_TEMP_DIR": str(ASSETS / "temp/gradio"),
    "MPLCONFIGDIR": str(ASSETS / "cache/matplotlib-ui"), "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "TEMP": str(ASSETS / "temp"), "TMP": str(ASSETS / "temp")}.items():
    os.environ[key] = value


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def tail(path, length=24000):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - length))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def folders():
    return sorted((folder for folder in EVIDENCE.iterdir() if folder.is_dir() and (folder / "result.json").is_file()), key=lambda folder: folder.name, reverse=True)


def selected_folder(name):
    folder = (EVIDENCE / (name or "")).resolve()
    if not name or not folder.is_relative_to(EVIDENCE.resolve()) or not (folder / "result.json").is_file():
        raise ValueError("Chọn một lượt chạy trong danh sách")
    return folder


def percent(value):
    return round(value * 100, 2) if isinstance(value, (float, int)) else None


def overview():
    table = []
    for folder in folders():
        result = read_json(folder / "result.json", {})
        if result.get("preflight_only"):
            continue
        sources = [(result.get("model_variant", ""), result.get("metrics"))]
        if result.get("comparison") and not result.get("metrics"):
            sources = [("Gốc", result["comparison"].get("baseline")), ("Sau GRPO", result["comparison"].get("after_static_grpo"))]
        for variant, metrics in sources:
            if not metrics:
                continue
            split = result.get("split") or ("Train" if folder.name.startswith("train-diagnostic-") else "Validation")
            variant = {"original_base": "Gốc", "base_plus_adapter": "Có adapter (xem log)"}.get(variant, variant)
            table.append([folder.name, variant, result.get("prompt_version", ""), split, metrics.get("generated_tests"), percent(metrics.get("valid_rate")),
                          percent(metrics.get("io_accuracy")), percent(metrics.get("attack_rate")),
                          str(metrics.get("cases_detected_with_up_to_4_tests")) + "/" + str(metrics.get("evaluated_cases"))])
    return table


def details(name):
    folder = selected_folder(name)
    return read_json(folder / "result.json", {}), tail(folder / "console.txt")


def curves(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from export_tensorboard import scalar_rows

    folder = selected_folder(name)
    series = scalar_rows(folder)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfa")
    sft = (folder / "sft-steps.jsonl").exists()
    for axis, tag, title, color in ((axes[0], "sft/loss" if sft else "rollout/reward_mean", "Loss SFT" if sft else "Reward trung bình", "#546d58"),
                                    (axes[1], "gradient/norm_observed", "Độ lớn gradient", "#7e8a76")):
        points = [(step, value) for key, step, value in series if key == tag]
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=1.8)
        else:
            axis.text(0.5, 0.5, "No training scalars in this run", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title, fontsize=12, color="#3d4936", loc="left", pad=16)
        axis.set_xlabel("Nhóm" if axis is axes[0] and not sft else "Bước cập nhật", fontsize=10, color="#8a9381")
        axis.set_facecolor("#fcfcfa")
        axis.tick_params(colors="#8a9381", labelsize=9, length=0, pad=8)
        axis.grid(axis="y", alpha=0.12)
        axis.spines[["left", "bottom"]].set_color("#e1e5d9")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    plt.close(figure)
    return figure


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.log = None
        self.session = None
        self.task = None
        self.stopped = False

    def start(self, task):
        if task not in TASKS:
            raise ValueError("Tác vụ không hợp lệ")
        return self.launch(task, SCRIPTS / TASKS[task])

    def start_demo(self, key, seed):
        from ui_demo import catalog
        data = catalog()
        if key not in data["cases"] or int(seed) not in (42, 43, 44):
            raise ValueError("Lựa chọn Demo không hợp lệ")
        pointer = read_json(PROJECT / "DEMO_CATALOG.json")
        return self.launch("Demo · " + data["cases"][key]["title"], SCRIPTS / "live_demo_worker.py",
                           {"case_key": key, "seed": int(seed), "catalog_path": pointer["path"], "catalog_sha256": pointer["sha256"]})

    def launch(self, task, target, request=None):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("Một tác vụ đang chạy. Hãy đợi tác vụ đó hoàn tất.")
            if not target.is_file() or not TRAIN_PYTHON.is_file():
                raise ValueError("Không tìm thấy Python hoặc chương trình của đồ án")
            self.session = EVIDENCE / "ui-jobs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self.session.mkdir(parents=True, exist_ok=False)
            self.log = self.session / "console.txt"
            command = [str(TRAIN_PYTHON), str(target)]
            if request is not None:
                shutil.copy2(request["catalog_path"], self.session / "catalog.json")
                if hashlib.sha256((self.session / "catalog.json").read_bytes()).hexdigest() != request["catalog_sha256"]:
                    raise ValueError("Dữ liệu Demo đã thay đổi")
                request["created_at"] = datetime.now().astimezone().isoformat()
                (self.session / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
                command += ["--root", str(self.session)]
            environment = os.environ.copy()
            environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HOME=str(ASSETS / "cache/huggingface"))
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with self.log.open("w", encoding="utf-8") as stream:
                self.process = subprocess.Popen(command, cwd=str(PROJECT), env=environment, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
            self.task, self.stopped = task, False
            import psutil
            (self.session / "launch.json").write_text(json.dumps({"task": task, "kind": "live_demo" if request else "experiment", "command": command, "pid": self.process.pid,
                "create_time": psutil.Process(self.process.pid).create_time(), "started_at": datetime.now().astimezone().isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.status()

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                import psutil
                parent = psutil.Process(self.process.pid)
                metadata = read_json(self.session / "launch.json")
                if abs(parent.create_time() - metadata["create_time"]) > 0.1 or parent.cmdline() != metadata["command"]:
                    raise RuntimeError("Tiến trình không còn khớp lượt đã khởi chạy; không tự động dừng.")
                processes = list(reversed(parent.children(recursive=True))) + [parent]
                for process in processes:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(processes, timeout=3)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(alive, timeout=3)
                if alive:
                    raise RuntimeError("Chưa dừng được tác vụ; kiểm tra tiến trình trước khi chạy lượt khác.")
                self.stopped = True
                (self.session / "stopped.json").write_text(json.dumps({"stopped_at": datetime.now().astimezone().isoformat(), "returncode": self.process.poll()}), encoding="utf-8")
        return self.status()

    def status(self):
        if self.process is None:
            return "Sẵn sàng — chưa có tác vụ được chạy từ giao diện.", ""
        code = self.process.poll()
        if code is None:
            state = "Đang chạy"
        elif self.stopped:
            state = "Đã yêu cầu dừng; chỉ kết quả đã ghi được giữ"
        elif code == 0:
            state = "Đã hoàn tất"
        else:
            state = "Có lỗi; xem log bên dưới (mã " + str(code) + ")"
        return state + " · " + self.task, tail(self.log)


JOBS = Jobs()


def build_ui():
    import gradio as gr
    from ui_presenter import workbench
    import ui_demo

    choices = [folder.name for folder in folders()]
    def training_choices():
        data = ui_demo.catalog()
        latest = [str(Path(data["variants"][key]["adapter_path"]).parent.relative_to(EVIDENCE)) for key in ("adversarial", "control", "parent")]
        pointer = read_json(PROJECT / "LATEST_GRPO_REPLICATES.json", {})
        replica_root = Path(pointer["path"]).resolve() if pointer.get("path") else None
        replicated = []
        if replica_root and replica_root.is_relative_to(EVIDENCE.resolve()) and read_json(replica_root / "result.json", {}).get("status") == "grpo_training_seed_replicates_completed":
            replicated = [str((replica_root / ("seed-" + str(seed)) / ("train-" + arm)).relative_to(EVIDENCE)) for seed in (43, 44) for arm in ("control", "adversarial")]
        older = [folder.name for folder in folders() if (folder / "rollouts.jsonl").exists() and "self-check" not in folder.name and not read_json(folder / "result.json", {}).get("preflight_only")]
        return list(dict.fromkeys(latest + replicated + older))
    training_runs = training_choices()
    def training_options():
        labels = ("GRPO đối kháng sau SFT · 96 bước", "GRPO cố định sau SFT · 96 bước", "SFT khởi động · 72 bước")
        choices = []
        for i, name in enumerate(training_choices()):
            parts = Path(name).parts
            label = labels[i] if i < 3 else name
            if len(parts) == 3 and parts[1] in ("seed-43", "seed-44") and parts[2] in ("train-control", "train-adversarial"):
                arm = "cố định" if parts[2] == "train-control" else "đối kháng"
                label = f"GRPO {arm} · seed train {parts[1][5:]} · 96 bước"
            choices.append((label, name))
        return choices
    training = training_runs[0] if training_runs else None
    initial, log, running = workbench(EVIDENCE, JOBS)
    with gr.Blocks(title="Sinh test đối kháng", analytics_enabled=False) as app:
        gr.HTML('<h1>Sinh test đối kháng</h1>', elem_id="app-header")
        with gr.Tab("Demo"):
            with gr.Row(elem_id="demo-selectors"):
                case = gr.Dropdown(ui_demo.choices(), value=ui_demo.EXAMPLES[0][1], label="Bài toán", scale=5)
                seed = gr.Radio([42, 43, 44], value=42, label="Seed sinh test", scale=1, min_width=220)
            with gr.Row(elem_id="example-buttons"):
                example_buttons = [gr.Button(label, size="sm") for label, _, _ in ui_demo.EXAMPLES]
            context = gr.HTML(ui_demo.context(ui_demo.EXAMPLES[0][1]))
            mode = gr.Radio(["Bản ghi đã lưu", "Chạy trên máy"], value="Bản ghi đã lưu", show_label=False, container=False)
            cached = gr.HTML(ui_demo.saved(ui_demo.EXAMPLES[0][1], 42))
            with gr.Group(visible=False, elem_id="live-demo-panel") as live_group:
                with gr.Row():
                    run_demo = gr.Button("Chạy ba model", variant="primary", elem_id="run-demo")
                    stop_demo = gr.Button("Dừng lượt chạy", interactive=False)
                live_result = gr.HTML(ui_demo.live_view(None))
            with gr.Accordion("Log chạy trực tiếp", open=False):
                demo_log = gr.Textbox(show_label=False, lines=10, interactive=False)

            def selection(key, value):
                return ui_demo.context(key), ui_demo.saved(key, int(value))
            case.change(selection, [case, seed], [context, cached])
            seed.change(selection, [case, seed], [context, cached])
            for button, (_, key, value) in zip(example_buttons, ui_demo.EXAMPLES):
                button.click(lambda k=key, s=value: (k, s), outputs=[case, seed])
            mode.change(lambda value: (gr.HTML(visible=value == "Bản ghi đã lưu"), gr.Group(visible=value == "Chạy trên máy")), mode, [cached, live_group])

            def refresh_demo(key, value):
                active = JOBS.process is not None and JOBS.process.poll() is None
                session = JOBS.session
                if session is None:
                    parent = EVIDENCE / "ui-jobs"
                    saved_runs = sorted(parent.glob("*/request.json"), reverse=True) if parent.exists() else []
                    session = saved_runs[0].parent if saved_runs else None
                request = read_json(session / "request.json", {}) if session else {}
                matches = request.get("case_key") == key and request.get("seed") == int(value)
                view = ui_demo.live_view(session if matches else None, active=active and session == JOBS.session)
                text = tail(session / "console.txt") if matches else ""
                return (view, text, gr.Button(interactive=not active), gr.Button(interactive=active), gr.Dropdown(interactive=not active), gr.Radio(interactive=not active),
                        *[gr.Button(interactive=not active) for _ in example_buttons])
            demo_outputs = [live_result, demo_log, run_demo, stop_demo, case, seed, *example_buttons]
            def launch_demo(key, value):
                try:
                    JOBS.start_demo(key, value)
                except Exception as error:
                    raise gr.Error(str(error)) from error
                return refresh_demo(key, value)
            run_demo.click(launch_demo, [case, seed], demo_outputs, concurrency_limit=1)
            stop_demo.click(lambda key, value: (JOBS.stop(), refresh_demo(key, value))[1], [case, seed], demo_outputs, queue=False)
            gr.Timer(2).tick(refresh_demo, [case, seed], demo_outputs, show_progress="hidden", queue=False)
        with gr.Tab("Kết quả"):
            metric_seed = gr.Radio(["Trung bình 3 seed", "42", "43", "44"], value="Trung bình 3 seed", label="Kết quả sinh test")
            summary = gr.HTML(ui_demo.comparison("Trung bình 3 seed"))
            metric_seed.change(ui_demo.comparison, metric_seed, summary)
            with gr.Accordion("Kết quả khi thay seed huấn luyện", open=False):
                stability = gr.HTML(ui_demo.training_stability(), elem_id="training-stability")
                refresh_stability = gr.Button("Cập nhật kết quả seed train", size="sm")
                refresh_stability.click(ui_demo.training_stability, outputs=stability)
            with gr.Accordion("Phạm vi và cấu hình", open=False):
                gr.Markdown("Hai nhánh GRPO bắt đầu từ cùng adapter SFT, mỗi nhánh 96 bước và 384 mẫu sinh. Nhánh đối kháng thay 20/96 lượt bằng 11 chương trình do adversary sinh. Adversary giữ nguyên trọng số; protagonist được cập nhật bằng reward.\n\nSau SFT kế thừa một lượt GRPO sơ bộ trước đó, vì vậy đây là so sánh ba mốc sau khởi động. Các bài toán là dữ liệu Python nhỏ tự xây dựng, chưa tái hiện benchmark lớn của bài báo.\n\nSeed huấn luyện: 42. Seed sinh test: 42, 43, 44. Chưa đánh giá tập Test. Các ví dụ trong Demo dùng để minh họa; bảng này tổng hợp toàn bộ Validation.")
            with gr.Accordion("Lịch sử cũ", open=False):
                gr.Dataframe(value=overview(), headers=["Lượt chạy", "Model", "Prompt", "Tập", "Số test", "Hợp lệ (%)", "Expected đúng (%)", "Test bắt lỗi (%)", "Phiên bản phát hiện"], interactive=False, wrap=True)
        with gr.Tab("Thực nghiệm"):
            with gr.Row(elem_id="workspace-row"):
                with gr.Column(scale=0, min_width=290, elem_id="controls"):
                    gr.Markdown("### Tác vụ")
                    task = gr.Dropdown(list(TASKS), value=next(iter(TASKS)), show_label=False, container=False)
                    start = gr.Button("Bắt đầu", variant="primary", elem_id="start-run", interactive=not running)
                    stop = gr.Button("Dừng", elem_id="stop-run", interactive=running)
                    gr.HTML('<div class="model-line"><span>Model</span><strong>Qwen3.5 · 2B</strong></div>', elem_id="model-caption")
                with gr.Column(scale=1, min_width=360, elem_id="workspace-column"):
                    progress = gr.HTML(initial, elem_id="workbench-view")
                    with gr.Accordion("Log chi tiết", open=False, elem_id="log-details"):
                        live = gr.Textbox(value=log, show_label=False, lines=12, max_lines=18, interactive=False, container=False)
            def refresh_job():
                value, text, active = workbench(EVIDENCE, JOBS)
                return value, text, gr.Button(interactive=not active), gr.Button(interactive=active)
            def start_job(value):
                try:
                    JOBS.start(value)
                    return refresh_job()
                except Exception as error:
                    raise gr.Error(str(error)) from error
            def stop_job():
                JOBS.stop()
                return refresh_job()
            outputs = [progress, live, start, stop]
            start.click(start_job, task, outputs, concurrency_limit=1)
            stop.click(stop_job, outputs=outputs, queue=False)
            gr.Timer(2).tick(refresh_job, outputs=outputs, show_progress="hidden", queue=False)
        with gr.Tab("Huấn luyện"):
            with gr.Row():
                selected = gr.Dropdown(training_options(), value=training, label="Lượt chạy", scale=4)
                refresh_choices = gr.Button("Cập nhật", scale=0)
            plot = gr.Plot(value=curves(training) if training else None, show_label=False, elem_id="training-plot")
            with gr.Accordion("Thông tin và log chi tiết", open=False):
                data = gr.JSON(label="Thông tin lượt chạy", value=details(training)[0] if training else {})
                console = gr.Textbox(label="Phần cuối log", lines=10, value=details(training)[1] if training else "", interactive=False)
            selected.change(lambda value: (curves(value), *details(value)), selected, [plot, data, console])
            refresh_choices.click(lambda: gr.Dropdown(choices=training_options()), outputs=selected)
        with gr.Tab("TensorBoard"):
            gr.HTML('<a href="http://127.0.0.1:6006" target="_blank" rel="noopener">Mở trong tab mới ↗</a>', elem_id="tensorboard-link")
            gr.HTML('<iframe title="TensorBoard" src="http://127.0.0.1:6006" style="width:100%;height:760px;border:1px solid #e0e5da;background:white"></iframe>')
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for folder in (ASSETS / "temp/gradio", ASSETS / "cache/matplotlib-ui"):
        folder.mkdir(parents=True, exist_ok=True)
    active = read_json(PROJECT / "configs/active-dataset.json", {})
    if active.get("dataset") == "apps-introductory":
        from apps_ui import CSS, build_ui as build_apps_ui
        app = build_apps_ui()
        stylesheet = CSS
    else:
        app = build_ui()
        stylesheet = Path(__file__).with_suffix(".css").read_text(encoding="utf-8")
    if args.check:
        print("UI BUILD CHECK PASSED. No model or training was started.")
        return
    import gradio as gr
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
               theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate", radius_size="sm", font=["Segoe UI", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"]),
               css=stylesheet,
               js="() => { document.documentElement.classList.remove('dark'); const b = document.getElementById('stop-run'); if (b) b.title = 'Dừng lượt đang chạy. Chỉ kết quả đã lưu được giữ.'; }", footer_links=[])


if __name__ == "__main__":
    main()
