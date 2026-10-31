import html
import json
from pathlib import Path

from apps_core import ASSETS, canonical, digest, read


def study():
    pointer = ASSETS / "evidence/APPS_EXPANDED_STUDY.json"
    if not pointer.exists():
        return None
    index = read(pointer)
    if not index.get("presentation_path"):
        return None
    path = Path(index["presentation_path"])
    if digest(path) != index["presentation_sha256"]:
        raise ValueError("Bảng kết quả đã thay đổi so với bản lưu.")
    return read(path)


def variants(current):
    report = study()
    if report:
        return report["variants"]
    chosen = current.get("selected_local_model")
    if chosen:
        return [{"key": "base", "label": "Qwen gốc", "adapter": None, "validation": current["baseline_validation"]},
                {"key": "sft", "label": "Sau SFT", "adapter": chosen["adapter_path"], "adapter_sha256": chosen["adapter_sha256"], "validation": chosen["validation"]}]
    return [{"key": "base", "label": "Qwen gốc", "adapter": None, "validation": current.get("baseline_validation")}]


def selected_key(current):
    report = study()
    return report["selected_variant"] if report else "sft" if current.get("selected_local_model") else "base"


def problem(case):
    number = case["case_id"].rsplit("/m", 1)[-1]
    return ('<article class="apps-case"><div class="apps-case-heading"><div><h2>' + html.escape(case["title"])
            + '</h2><p class="apps-muted">Phiên bản lỗi ' + html.escape(number)
            + ' · <a href="' + html.escape(case["source_url"], quote=True) + '" target="_blank" rel="noopener">Đề APPS gốc</a></p></div></div>'
            + '<div class="apps-problem-grid"><div><h4>Code cần kiểm tra</h4><pre>' + html.escape(case["buggy_code"])
            + '</pre></div><div><h4>Miền input thực nghiệm</h4><pre>' + html.escape(input_contract(case))
            + '</pre></div></div><details><summary>Xem đề bài đầy đủ</summary><pre>'
            + html.escape(case["specification"]) + '</pre></details></article>')


def input_contract(case):
    lines = []
    for name, field in case["domain"].items():
        if "enum" in field:
            rule = "một trong các giá trị " + canonical(field["enum"])
        elif field["type"] == "int":
            rule = f'số nguyên từ {field["min"]} đến {field["max"]}'
        else:
            kind = "chuỗi" if field["type"] == "str" else "danh sách"
            unit = "ký tự" if field["type"] == "str" else "phần tử"
            rule = f'{kind} dài {field["min_length"]}–{field["max_length"]} {unit}, lấy từ {canonical(field["alphabet"])}'
        lines.append(name + ": " + rule + ".")
    for constraint in case.get("constraints", []):
        kind, *names = constraint.split(":")
        if kind == "lt":
            lines.append(names[0] + " < " + names[1] + ".")
        elif kind == "same_length":
            lines.append(names[0] + " và " + names[1] + " có cùng độ dài.")
        elif kind == "not_all_zero":
            lines.append(names[0] + " có ít nhất một phần tử khác 0.")
    return "\n".join(lines)


def result_card(title, row):
    output = '<section class="apps-result"><h3>' + html.escape(title) + '</h3>'
    if not row:
        return output + '<p class="apps-muted">Chưa có kết quả.</p></section>'
    samples = row["samples"]
    hits = sum(bool(s["bug_detected"]) for s in samples)
    correct = sum(bool(s["correct_expected"]) for s in samples)
    output += f'<p class="apps-result-count">{hits}/{len(samples)} test phát hiện lỗi <span>· {correct}/{len(samples)} expected đúng</span></p>'
    output += '<div class="apps-table-wrap"><table class="apps-test-table"><thead><tr><th>Test / Input</th><th>Expected</th><th>Code trả về</th></tr></thead><tbody>'
    for index, sample in enumerate(samples, 1):
        value = sample.get("test")
        if not value:
            try:
                value = json.loads(sample["raw_output"])
            except (ValueError, TypeError):
                value = None
        has_input = type(value) is dict and "input" in value
        text = canonical(value["input"]) if has_input else sample["raw_output"]
        label = "Phát hiện lỗi" if sample["bug_detected"] else "Expected sai" if sample["valid_format"] and not sample["correct_expected"] else "Chưa phát hiện lỗi" if sample["correct_expected"] else "Sai miền / cấu trúc" if sample.get("error") == "schema_or_domain" else "Đầu ra bị cắt" if sample.get("error") == "truncated" else "JSON không hợp lệ"
        tone = "apps-hit" if sample["bug_detected"] else "apps-neutral" if sample["correct_expected"] else "apps-wrong"
        expected = canonical(value["expected"]) if type(value) is dict and "expected" in value else "—"
        observed = sample.get("buggy_error", "—")
        if "buggy_output" in sample:
            observed = canonical(sample["buggy_output"])
        output += '<tr><td><span class="apps-test-index">' + str(index) + '</span><code>' + html.escape(text) + '</code><span class="apps-test-status ' + tone + '">' + label + '</span></td><td><code>' + html.escape(expected) + '</code>'
        if sample.get("valid_format") and not sample.get("correct_expected"):
            output += '<small>Đúng: <code>' + html.escape(canonical(sample.get("gold_output"))) + '</code></small>'
        output += '</td><td><code>' + html.escape(observed) + '</code></td></tr>'
    output += '</tbody></table></div><details class="apps-raw"><summary>JSON gốc</summary><pre>'
    output += html.escape('\n'.join(f'Test {i}: {s["raw_output"]}' for i, s in enumerate(samples, 1)))
    return output + '</pre></details></section>'


def comparison(current, identity, key):
    versions = variants(current)
    keys = ("base", key) if key != "base" else ("base",)
    cards = []
    for item in versions:
        if item["key"] not in keys:
            continue
        record = item.get("validation")
        row = None
        if record:
            folder = Path(record["path"])
            if digest(folder / "result.json") != record["result_sha256"]:
                raise ValueError("Kết quả đã lưu bị thay đổi.")
            rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            row = next((row for row in rows if row["case_id"] == identity), None)
        cards.append(result_card(item["label"], row))
    return '<div class="apps-compare">' + ''.join(cards) + '</div>'


def metric_rows(report, split):
    summary = report[split + "_summary"]
    total = report[split + "_cases"]
    labels = {v["key"]: v["label"] for v in report["variants"]}
    output = []
    for version in report["variants"]:
        name = version["key"]
        values = summary[name]
        output.append([labels[name], f'{values["valid_tests"]["mean"] / (total * 4) * 100:.1f}',
                       f'{values["correct_expected"]["mean"] / (total * 4) * 100:.1f}',
                       f'{values["bug_detecting_tests"]["mean"] / (total * 4) * 100:.1f}',
                       f'{values["cases_detected"]["mean"]:.2f} / {total}'])
    return output


def summary_table(report, split):
    values = report[split + "_summary"]
    keys = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
    output = '<div class="apps-table-wrap"><table class="apps-summary-table"><thead><tr><th>Phiên bản</th><th>Hợp lệ</th><th>Expected đúng</th><th>Test bắt lỗi</th><th>Code phát hiện @4</th></tr></thead><tbody>'
    for row, version in zip(metric_rows(report, split), report["variants"], strict=True):
        key = version["key"]
        tone = ' class="apps-selected-row"' if key == report["selected_variant"] else ""
        output += '<tr' + tone + '><td>' + html.escape(row[0]) + '</td>'
        for field, text in zip(keys, row[1:], strict=True):
            label = html.escape(text + ("%" if field != "cases_detected" else ""))
            improved = key != "base" and values[key][field]["mean"] > values["base"][field]["mean"]
            output += '<td>' + ('<strong>' + label + '</strong>' if improved else label) + '</td>'
        output += '</tr>'
    return output + '</tbody></table></div>'
