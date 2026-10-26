import functools
import hashlib
import html
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ("parent", "control", "adversarial")
EXAMPLES = [
    ("Đảo danh sách · cải thiện sau RL", "validation|p17-reverse/m1", 42),
    ("Chuỗi đối xứng · đối kháng bắt được lỗi", "validation|p21-palindrome/m1", 44),
    ("Loại trùng · kết quả giảm sau RL", "validation|p16-unique-sorted/m1", 42),
    ("Chuỗi đối xứng · cả ba chưa bắt được lỗi", "validation|p21-palindrome/m2", 42),
]
LABELS = {"bug_detected": "Bắt được lỗi", "correct_but_no_bug": "Đúng, chưa bắt lỗi", "wrong_expected": "Expected sai",
          "invalid_json": "JSON không hợp lệ", "invalid_schema": "Sai cấu trúc JSON", "input_outside_domain": "Input ngoài miền", "truncated": "Sinh chưa hoàn chỉnh"}


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
        raise ValueError("Dữ liệu Demo đã thay đổi; cần tạo lại danh mục từ log gốc.")
    return json.loads(raw)


def catalog():
    pointer = read(PROJECT / "DEMO_CATALOG.json")
    return _load(pointer["path"], pointer["sha256"])


def choices():
    data = catalog()
    preferred = [key for _, key, _ in EXAMPLES]
    keys = preferred + [key for key in data["cases"] if key not in preferred]
    return [(data["cases"][key]["title"] + " · " + data["cases"][key]["case_id"].split("/")[-1] + (" · Train đối kháng" if key.startswith("pool|") else ""), key) for key in keys]


def domain_text(domain):
    result = []
    for name, spec in domain.items():
        kind = spec["type"]
        if kind == "int":
            value = f'số nguyên từ {spec["min"]} đến {spec["max"]}'
        elif kind == "list":
            value = f'danh sách dài {spec["min_length"]}–{spec["max_length"]}, phần tử thuộc {compact(spec["alphabet"])}'
        else:
            value = f'chuỗi dài {spec["min_length"]}–{spec["max_length"]}, ký tự thuộc {compact(spec["alphabet"])}'
        result.append(f'<li><code>{esc(name)}</code>: {esc(value)}</li>')
    return "".join(result)


def context(key):
    case = catalog()["cases"][key]
    subset = "Validation · chưa dùng để cập nhật trọng số" if case["subset"] == "validation" else "Train · code do adversary sinh"
    return f'<section class="demo-context"><div><div class="eyebrow">{subset}</div><h2>{esc(case["title"])}</h2><p>{esc(case["specification_vi"])}</p><ul>{domain_text(case["domain"])}</ul></div><div><div class="section-label">Code cần kiểm tra</div><pre><code>{esc(case["buggy_code"])}</code></pre></div></section>'


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
        body = f'<div class="test-input"><span>Input</span><code>{esc(compact(test["input"]))}</code></div><div class="test-values"><span>Expected của model</span><code>{esc(compact(test["expected"]))}</code></div>'
        if sample.get("valid_format"):
            body += f'<div class="test-values"><span>Kết quả đúng</span><code>{esc(compact(sample.get("gold_output")))}</code></div>'
            if "buggy_output" in sample or "buggy_error" in sample:
                value = sample.get("buggy_error") or compact(sample.get("buggy_output"))
                body += f'<div class="test-values"><span>Code trả về</span><code>{esc(value)}</code></div>'
    else:
        body = '<div class="invalid-copy">Không có cặp input/expected hoàn chỉnh.</div>'
    raw = esc(sample["raw_output"])
    return f'<article class="test-sample {tone}"><header><span>Test {index}</span><strong>{LABELS[reason]}</strong></header>{body}<details><summary>Đầu ra gốc</summary><pre>{raw}</pre></details></article>'


def render_records(key, records, seed, mode, stamp=""):
    data = catalog()
    case = data["cases"][key]
    cards = ""
    for variant in VARIANTS:
        record = records.get(variant)
        label = data["variants"][variant]["label"]
        if not record:
            cards += f'<section class="model-tests"><h3>{label}</h3><p class="empty-copy">Chưa có kết quả.</p></section>'
            continue
        samples = record["samples"]
        valid = sum(bool(s["valid_format"]) for s in samples)
        correct = sum(bool(s["correct_expected"]) for s in samples)
        hits = sum(bool(s["bug_detected"]) for s in samples)
        cards += f'<section class="model-tests"><div class="model-tests-head"><h3>{label}</h3><strong class="{"hit-text" if hits else "muted-text"}">{hits}/4 test bắt lỗi</strong></div><p class="model-counts">{valid}/4 hợp lệ · {correct}/4 expected đúng</p>' + "".join(sample_html(s, case, i) for i, s in enumerate(samples, 1)) + '</section>'
    return f'<div class="demo-provenance"><strong>{esc(mode)}</strong><span>Seed {seed} · 4 test / model{(" · " + esc(stamp)) if stamp else ""}</span></div><div class="model-grid">{cards}</div>'


def saved(key, seed):
    data = catalog()
    values = data["predictions"][key]
    if str(seed) not in values:
        return '<div class="empty-copy">Tập Train đối kháng hiện chỉ có bản ghi seed 42. Chọn seed 42 hoặc chạy trực tiếp.</div>'
    return render_records(key, values[str(seed)], seed, "BẢN GHI ĐÃ LƯU")


def comparison(selection):
    data = catalog()
    mean = selection == "Trung bình 3 seed"
    metrics = data["experiment"]["decoding_seed_mean_rates"] if mean else data["metrics"]["validation"][str(selection)]
    definitions = [("Test hợp lệ", "valid_rate", "valid_tests", "generated_tests"), ("Expected đúng", "io_accuracy", "correct_expected", "generated_tests"),
                   ("Test bắt lỗi", "attack_rate", "bug_detecting_tests", "generated_tests"), ("Phiên bản lỗi phát hiện @4", "empirical_case_detection_at_4", "cases_detected_with_up_to_4_tests", "evaluated_cases")]
    body = ""
    for title, rate, numerator, denominator in definitions:
        body += '<tr><th>' + title + '</th>'
        for variant in VARIANTS:
            value = metrics[variant][rate] * 100
            count = "Trung bình 3 seed" if mean else f'{metrics[variant][numerator]}/{metrics[variant][denominator]}'
            body += f'<td><strong>{value:.2f}%</strong><small>{count}</small><div class="mini-track"><b style="width:{value:.4f}%"></b></div></td>'
        body += '</tr>'
    return '<section class="comparison"><div class="comparison-heading"><h2>So sánh trên Validation</h2><span>6 bài · 18 phiên bản lỗi</span></div><div class="table-scroll"><table><thead><tr><th>Chỉ số</th><th>Sau SFT</th><th>GRPO cố định</th><th>GRPO đối kháng</th></tr></thead><tbody>' + body + '</tbody></table></div><p class="comparison-meta">Mỗi seed: 72 test / model. Ba tỷ lệ đầu tính trên toàn bộ test sinh, kể cả test sai. @4 dùng tối đa 4 test cho mỗi phiên bản lỗi.</p><p class="result-limit">Một seed huấn luyện; ba seed sinh test (42, 43, 44). Kết quả trên dữ liệu nội bộ nhỏ, chưa đánh giá tập Test.</p></section>'


def training_stability():
    pointer = PROJECT / "LATEST_GRPO_REPLICATES.json"
    if not pointer.is_file():
        return '<p class="empty-copy">Thí nghiệm thay seed huấn luyện chưa hoàn tất. Theo dõi lượt đang chạy tại Thực nghiệm.</p>'
    metadata = read(pointer)
    root = Path(metadata["path"]).resolve()
    evidence = Path(r"D:\DA\TTNT_De19_Assets\evidence").resolve()
    if not root.is_relative_to(evidence):
        raise ValueError("Invalid replication result path")
    result = read(root / "result.json")
    if result.get("status") != "grpo_training_seed_replicates_completed":
        return '<p class="empty-copy">Thí nghiệm chưa hoàn tất.</p>'
    lines = ""
    for seed in (42, 43, 44):
        values = result["all_metrics"][str(seed)]
        a, b = (values[arm]["cases_detected_with_up_to_4_tests"] for arm in ("control", "adversarial"))
        lines += f'<tr><th>{seed}</th><td><strong>{a}/18</strong></td><td><strong>{b}/18</strong></td><td>{b-a:+d} code</td></tr>'
    means = result["aggregate"]
    a, b = (means[arm]["empirical_case_detection_at_4"] for arm in ("control", "adversarial"))
    lines += f'<tr><th>Trung bình ± SD</th><td><strong>{100*a["mean"]:.2f}%</strong><small>± {100*a["sample_std"]:.2f} đpt</small></td><td><strong>{100*b["mean"]:.2f}%</strong><small>± {100*b["sample_std"]:.2f} đpt</small></td><td>{100*(b["mean"]-a["mean"]):+.2f} đpt</td></tr>'
    return '<section class="comparison"><div class="comparison-heading"><h2>Thay seed khi huấn luyện GRPO</h2><span>Seed sinh test cố định 42</span></div><div class="table-scroll"><table><thead><tr><th>Seed train</th><th>GRPO cố định</th><th>GRPO đối kháng</th><th>Chênh lệch</th></tr></thead><tbody>' + lines + '</tbody></table></div><p class="comparison-meta">Số code lỗi được phát hiện bằng tối đa 4 test, trên cùng 18 code Validation. SD là độ lệch chuẩn giữa các seed train, không phải khoảng tin cậy.</p><p class="result-limit">Cùng checkpoint SFT và cùng pool đối kháng; mỗi nhánh 96 bước từ đầu với bộ tối ưu mới. Đây là độ nhạy seed của GRPO, chưa phải ba lượt SFT + adversary + GRPO độc lập. Các model Demo vẫn dùng seed train 42.</p></section>'


def live_view(session, active=False):
    if session is None or not (session / "request.json").is_file():
        return '<div class="empty-copy">Chọn bài và bấm Chạy ba model. Model nền được nạp một lần; ba adapter sinh test lần lượt.</div>'
    request = read(session / "request.json")
    records = {}
    for variant in VARIANTS:
        path = session / variant / "prediction.json"
        if path.is_file():
            records[variant] = read(path)
    result_path = session / "result.json"
    result = read(result_path) if result_path.exists() else {}
    state = {"completed": "Đã chạy xong", "failed": "Có lỗi; mở Log chi tiết", "started": "Đang nạp / sinh test" if active else "Lượt chưa hoàn tất"}.get(result.get("status"), "Đang xử lý" if active else "Lượt chưa hoàn tất")
    if (session / "stopped.json").is_file():
        state = "Đã dừng; chỉ phần đã ghi được giữ"
    if result.get("error_category") == "memory":
        state = "Thiếu bộ nhớ. Đóng bớt ứng dụng rồi thử lại; kết quả đã sinh vẫn được giữ."
    stamp = result.get("finished_at", request["created_at"])
    return f'<p class="live-state">{state} · {len(records)}/3 model</p>' + render_records(request["case_key"], records, request["seed"], "CHẠY TRỰC TIẾP TRÊN MÁY", stamp)
