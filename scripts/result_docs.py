import csv
import json
from pathlib import Path


FIELDS = ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")
HEADERS = ("Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code phát hiện /15")
LABELS = {
    "base": "Qwen gốc", "baseline_validation": "Qwen gốc", "fresh_grpo_pilot": "GRPO ban đầu", "sft": "SFT64", "sft256": "SFT256",
    "sft256-grpo96": "SFT256 + GRPO96", "grpo": "GRPO", "control": "GRPO cố định", "adaptive": "GRPO đối kháng",
    "sft_validation": "SFT64", "pilot_validation": "SFT64 + GRPO96", "SFT64_parent": "SFT64",
    "legacy-v1": "Reward 0,1/0,2/0,7", "equal-v1": "Reward chia đều",
    "attention-qv-v1": "LoRA 6 khối", "hybrid-qkv-v1": "LoRA 24 khối",
    "SFT64_historical": "SFT64 trước đó", "SFT64_same_trajectory": "SFT64 cùng lượt", "SFT256": "SFT256",
}
NOTES = {
    "apps-pilot": ("Thử nghiệm APPS ban đầu", "Prompt v1, Qwen gốc và lượt GRPO đầu tiên. Mỗi code Validation sinh bốn test, seed 42.", "Đây là mốc đầu trước khi đổi prompt và thêm SFT.", []),
    "apps-sft-grpo": ("SFT và GRPO với nhãn expected-first", "So sánh Qwen gốc, SFT64 và GRPO96 trên 15 code Validation, seed 42. Nhãn SFT ở lượt này đặt expected trước input.", "Tỷ lệ hợp lệ tăng sau học; số code phát hiện vẫn là 3/15.", ["validation-comparison.png"]),
    "apps-adversarial": ("Curriculum đối kháng với nhãn expected-first", "Hai nhánh dùng cùng SFT64 cha, cùng hai vòng × 48 bước GRPO. Bảng dùng seed sinh 42.", "Cố định và đối kháng đều phát hiện 2/15 code ở mốc cuối.", ["validation-comparison.png"]),
    "apps-adversarial-review": ("Lấy mẫu lại hai nhánh expected-first", "Đánh giá bốn checkpoint trên các seed sinh 42/43/44. Mỗi checkpoint được huấn luyện một lần.", "Cố định phát hiện trung bình 2,67/15 code; đối kháng 3/15. Biến động giữa các seed còn lớn.", ["sampling.png"]),
    "apps-input-first-adversarial": ("Curriculum đối kháng với nhãn input-first", "Hai nhánh bắt đầu từ SFT64 input-first, cùng hai vòng × 48 bước GRPO. Bảng dùng seed sinh 42.", "Hai nhánh cùng phát hiện 3/15 code ở seed 42.", ["validation-comparison.png"]),
    "apps-input-first-review": ("Lấy mẫu lại hai nhánh input-first", "So sánh Qwen gốc, SFT64 và hai nhánh GRPO trên ba seed sinh 42/43/44.", "Cố định và đối kháng cùng đạt trung bình 4/15 code và 5/60 test bắt lỗi. Nhánh đối kháng cần thêm thời gian sinh và lọc code.", ["sampling.png"]),
    "apps-target-order": ("Thứ tự trường trong nhãn SFT", "Giữ nguyên 319 cặp input/expected và cấu hình học; đổi thứ tự hai trường ngoài cùng của target. So riêng các mốc SFT và GRPO.", "Input-first tăng expected ở cả ba seed sinh. Phần lớn thay đổi xuất hiện ngay sau SFT; mức tăng test bắt lỗi tập trung ở bài làm tròn 2947.", ["comparison.png"]),
    "apps-shared-sft-budget": ("Ngân sách SFT với nhãn input-first", "Checkpoint 64 và 256 được lấy từ cùng một lượt học, chung model khởi đầu và 64 bước đầu. Bảng Validation dùng ba seed sinh.", "Expected trung bình tăng 19 lên 26/60; test bắt lỗi 5,33 lên 10,33/60; code phát hiện 4 lên 7/15.", ["learning-curve.png"]),
    "apps-sft256-grpo": ("GRPO sau SFT256", "Thêm 96 bước GRPO từ đúng adapter SFT256, trên code Train cố định. So với parent trên cùng ba seed sinh.", "GRPO tăng nhẹ số test hợp lệ nhưng giảm expected, test bắt lỗi và code phát hiện. SFT256 được giữ làm checkpoint tốt hơn trong cặp này.", ["comparison.png"]),
    "apps-sft256-equal-reward": ("GRPO với reward chia đều sau SFT256", "Từ cùng adapter SFT256, học 96 bước GRPO với trọng số định dạng, expected và attack cùng bằng 1/3. Các điều kiện khác giữ như lượt reward mặc định.", "", ["comparison.png"]),
    "apps-sft-budget": ("Ngân sách SFT với nhãn expected-first", "So sánh 64 và 256 bước SFT với cùng pool nhãn và prompt. Có checkpoint 64 của chính lượt học mới để đối chiếu.", "SFT256 tăng expected nhưng giảm số code phát hiện. Kết quả này thuộc cấu hình nhãn cũ.", ["validation-comparison.png", "learning-curve.png"]),
    "apps-reward-ablation": ("Tỷ trọng reward trong GRPO", "Hai nhánh cùng SFT64 expected-first, cùng 96 bước GRPO; so trọng số 0,1/0,2/0,7 với chia đều 1/3.", "Reward chia đều tăng nhẹ ở seed 42, nhưng mức tăng chưa ổn định khi sinh lại với seed 43/44.", ["validation-comparison.png", "training-components.png"]),
    "apps-error-analysis": ("Phân tích lỗi ở lượt đối kháng ban đầu", "Phân loại đầu ra Validation của các checkpoint expected-first và đối chiếu những input có thể làm lộ lỗi.", "Expected sai làm mất một phần cơ hội bắt lỗi, kể cả khi input đã phân biệt được reference với mutant.", ["failure-breakdown.png"]),
    "apps-expected-diagnosis": ("Phân tích expected sai", "Đếm lỗi trên ba seed sinh của Qwen gốc, SFT64 input-first và các nhánh GRPO.", "Ở SFT64 + GRPO96, có 69/180 mẫu hợp lệ nhưng expected sai; 36 input trong số đó có thể làm lộ lỗi. Dữ liệu từng bài nằm trong per-problem.csv.", ["expected-errors.png"]),
    "apps-serialization-diagnosis": ("Thứ tự khóa JSON sau SFT", "Các nhãn cũ đều đặt expected trước input do hàm canonical sắp xếp tên khóa.", "Sau SFT, phần lớn JSON model sinh cũng bắt đầu bằng expected. Phép so sánh thứ tự target được lưu riêng tại apps-target-order.", []),
    "apps-input-difficulty": ("Phân bố input làm lộ lỗi", "Vét cạn miền input Validation để đo tỷ lệ input phân biệt được reference với từng mutant.", "Xác suất phát hiện với bốn lần chọn đều có hoàn lại là 1−(1−p)^4, giả sử expected luôn đúng. Đây là phân tích bằng reference, không phải thành tích của model.", []),
    "apps-lora-coverage": ("Phạm vi adapter LoRA", "So profile q_proj/v_proj của self-attention với profile thêm projection của linear attention.", "Profile cũ có 208.896 tham số trên 6/24 khối; profile mở rộng có 798.720 tham số trên 24/24 khối. Hai lượt SFT64 hoàn tất; phép thử dừng trước Validation để kiểm tra thứ tự nhãn.", []),
    "apps-lora-input-first": ("Phạm vi LoRA với nhãn input-first", "Giữ cùng 319 nhãn input-first, thứ tự học và 256 bước SFT. LoRA 6 khối có 208.896 tham số; LoRA 24 khối có 798.720 tham số.", "", ["train-comparison.png", "learning-curve.png"]),
    "apps-expected-budget": ("Lượt SFT dừng ở kiểm tra replay", "Chạy lại mốc 64 để kiểm tra điều kiện so ngân sách với adapter lịch sử.", "Lượt này dừng sau 116/256 bước. Relative L2 tại mốc 64 là 0,002685916, vượt ngưỡng 0,001. Nhãn, thứ tự ví dụ và mã dùng chung trùng hash; nguyên nhân độ lệch chưa được xác định. Không có đánh giá chất lượng mới trong lượt này.", []),
    "apps-tuning": ("Thinking và hợp đồng đầu ra", "Các phép thử suy luận trên Train với adapter đóng băng: bật thinking và thay diễn đạt kiểu đầu ra trong prompt.", "Nhánh thinking lưu 20 mẫu đều bị cắt ở trần 768 token. Hai prompt v3/v4 cũng không qua điều kiện Train; không chấm thêm Validation cho các ứng viên này.", []),
    "apps-data-profile": ("Thống kê dữ liệu", "Tập con APPS gồm 29 bài, 74 mutant, chia theo bài thành Train/Validation/Test. Phần thống kê đặc trưng dùng Train; số lượng tập chia lấy từ manifest.", "Dữ liệu nguồn, hợp đồng input và bộ chấm được mô tả trong tài liệu dữ liệu.", ["dataset-profile.png", "pipeline.png"]),
    "apps-final-test": ("Đánh giá Test", "Đánh giá các checkpoint và cấu hình đã chốt trên tập Test phát triển.", "Kết quả được tổng hợp từ các lượt đánh giá đã lưu.", ["comparison.png"]),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label(value):
    return LABELS.get(str(value), str(value))


def number(value):
    if isinstance(value, dict) and "mean" in value:
        return f'{value["mean"]:.2f} ± {value["sample_sd"]:.2f}'.replace(".", ",")
    return str(value).replace("|", "\\|")


def table(headers, rows, highlight=False):
    best = {i: max(row[i]["mean"] for row in rows) for i in range(1, len(headers))} if highlight else {}
    def cell(value, index):
        text = number(value)
        return "**" + text + "**" if index in best and value["mean"] == best[index] else text
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(cell(v, i) for i, v in enumerate(row)) + " |" for row in rows])


def csv_table(path, metric_headers=HEADERS):
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        return ""
    headers = records[0]
    valid = "valid_tests" if "valid_tests" in headers else "valid"
    bug = "bug_detecting_tests" if "bug_detecting_tests" in headers else "bug_tests"
    if not {valid, "correct_expected", bug, "cases_detected"}.issubset(headers):
        return ""
    keys = [k for k in ("variant", "profile", "sft_steps", "order", "stage", "generation_seed") if k in headers]
    titles = {"variant": "Cấu hình", "profile": "LoRA", "sft_steps": "Bước SFT", "order": "Nhãn", "stage": "Mốc", "generation_seed": "Seed sinh"}
    return table([titles[k] for k in keys] + list(metric_headers),
                 [[label(row[k]) for k in keys] + [row[f] for f in (valid, "correct_expected", bug, "cases_detected")] for row in records])


def summary_table(data, metric_headers=HEADERS):
    summary = data.get("summary", {})
    if not summary:
        return ""
    rows = []
    for name, fields in summary.items():
        if "correct_expected" in fields:
            rows.append([label(name), *[fields[f] for f in FIELDS]])
        else:
            for stage, values in fields.items():
                if isinstance(values, dict) and "correct_expected" in values:
                    rows.append([name + " / " + stage, *[values[f] for f in FIELDS]])
    return table(["Cấu hình", *metric_headers], rows, highlight=True) if rows else ""


def write_result_readme(folder):
    folder = Path(folder)
    title, intro, conclusion, figures = NOTES.get(folder.name, ("Kết quả thực nghiệm", "Số liệu và cấu hình của lượt chạy.", "", []))
    metrics = folder / "metrics.json"
    data = read(metrics) if metrics.exists() else {}
    headers = HEADERS
    if folder.name == "apps-final-test" and data:
        slots, cases = data["test_slots_per_run"], data["test_cases"]
        headers = (f"Hợp lệ /{slots}", f"Expected đúng /{slots}", f"Test bắt lỗi /{slots}", f"Code phát hiện /{cases}")
        selected = label(data["selected_variant_before_test"])
        intro = f'Test phát triển gồm {cases} code, mỗi lượt có {slots} test. Checkpoint được chọn từ Validation trước đợt đánh giá: {selected}.'
        conclusion = 'Đây là tập giữ lại từ APPS Train; APPS Test chính thức được ghi riêng trong protocol.json. Bảng trình bày trung bình và SD của các seed sinh, với một lượt train cho mỗi cấu hình.'
    parts = ["# " + title, "", intro, ""]
    if folder.name == "apps-lora-input-first":
        parts += ["## Train", "", csv_table(folder / "train.csv", ("Hợp lệ /164", "Expected đúng /164", "Test bắt lỗi /164", "Code phát hiện /41")), ""]
        if data.get("summary"):
            parts += ["## Validation", ""]
            figures = [*figures, "validation-comparison.png"]
            a, b = data["summary"]["attention-qv-v1"], data["summary"]["hybrid-qkv-v1"]
            values = [f'{row[field]["mean"]:.2f}'.replace(".", ",") for field in ("correct_expected", "bug_detecting_tests", "cases_detected") for row in (a, b)]
            conclusion = f'Expected trung bình: {values[0]} → {values[1]}/60; test bắt lỗi: {values[2]} → {values[3]}/60; code phát hiện: {values[4]} → {values[5]}/15. '
            conclusion += 'Điều kiện đặt trước là tăng ít nhất 9 expected đúng trên tổng 180 test, đồng thời các số đếm còn lại không giảm. Kết quả ' + ("đạt điều kiện này." if data["eligible_for_followup"] else "chưa đạt điều kiện này; giữ checkpoint 6 khối.")
        else:
            conclusion = 'LoRA mở rộng chưa qua điều kiện trên Train, nên lượt này dừng trước Validation. Checkpoint của demo được giữ nguyên.'
        conclusion += ' Đối chứng 6 khối dùng lại lượt SFT256 đã lưu; ứng viên 24 khối được học mới. Đây là phép thử phạm vi LoRA, với một seed train cho mỗi cấu hình.'
    if metrics.exists():
        summary = data.get("summary", {})
        if folder.name in ("apps-sft256-grpo", "apps-sft256-equal-reward"):
            winner = label(data["selected_within_this_pair"])
            conclusion = f'Trong cặp so sánh này, {winner} có kết quả tốt hơn theo tiêu chí chọn checkpoint. Các số đếm trước và sau GRPO được giữ riêng trong bảng.'
            if folder.name == "apps-sft256-equal-reward":
                a, b = summary["sft256"], summary["sft256-grpo96"]
                changes = {f: b[f]["mean"] - a[f]["mean"] for f in FIELDS}
                if changes["cases_detected"] == 0 and changes["bug_detecting_tests"] > 0 and changes["correct_expected"] < 0:
                    extra = round(changes["bug_detecting_tests"] * 3)
                    conclusion = f'Quy tắc xếp hạng ưu tiên số code rồi số test bắt lỗi xếp nhánh RL trước nhờ thêm {extra} test bắt lỗi trên tổng 180 test. Số code phát hiện trung bình bằng SFT, còn expected giảm; mức chênh lệch này chưa cho thấy chất lượng khái quát tốt hơn.'
        elif folder.name == "apps-shared-sft-budget" and summary:
            a, b = summary["64"], summary["256"]
            conclusion = f'Expected trung bình ở SFT64 và SFT256 lần lượt là {a["correct_expected"]["mean"]:.2f} và {b["correct_expected"]["mean"]:.2f}/60; số code phát hiện là {a["cases_detected"]["mean"]:.2f} và {b["cases_detected"]["mean"]:.2f}/15.'
        elif folder.name in ("apps-adversarial-review", "apps-input-first-review"):
            a, b = summary["control"], summary["adaptive"]
            conclusion = f'Nhánh cố định phát hiện trung bình {a["cases_detected"]["mean"]:.2f}/15 code; nhánh đối kháng {b["cases_detected"]["mean"]:.2f}/15. Chi phí sinh và lọc code được lưu trong costs.csv.'
        elif folder.name == "apps-lora-coverage" and summary:
            conclusion = 'So sánh các cấu hình LoRA trên cùng ngân sách và các seed sinh. Tham số, chi phí và kết quả từng lượt nằm trong các tệp dữ liệu bên dưới.'
        elif folder.name == "apps-expected-budget" and data.get("result", {}).get("status") != "stopped_at_replay_gate":
            title = "Ngân sách SFT"
            parts[0] = "# " + title
            conclusion = f'Trạng thái: {data["result"]["status"]}. Kết quả đánh giá, nếu có, được giữ riêng theo checkpoint và seed sinh.'
        text = summary_table(data, headers)
        if text:
            parts += [text, "", "Trung bình ± SD trên ba seed sinh; mỗi cấu hình có một lượt train.", ""]
    if (folder / "metrics.csv").exists():
        details = csv_table(folder / "metrics.csv", headers)
        if details:
            parts += ["## Kết quả từng lượt", "", details, ""]
    if folder.name == "apps-expected-diagnosis":
        counts = folder / "counts.csv"
        with counts.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        parts += [table(["Checkpoint", "Expected đúng /180", "Hợp lệ, expected sai /180", "Input lộ lỗi nhưng expected sai"],
                        [[label(r["variant"]), r["correct_expected"], r["wrong_expected"], r["input_exposes_bug_but_expected_wrong"]] for r in records]), ""]
    if folder.name == "apps-lora-input-first" and data.get("summary") and (folder / "per-problem.csv").exists():
        grouped = {}
        with (folder / "per-problem.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != "validation":
                    continue
                key = row["source_id"], row["variant"]
                values = grouped.setdefault(key, [0, 0, 0])
                for i, field in enumerate(("correct_expected", "bug_tests", "samples")):
                    values[i] += int(row[field])
        rows = []
        for problem in sorted({key[0] for key in grouped}):
            a, b = grouped[problem, "attention-qv-v1"], grouped[problem, "hybrid-qkv-v1"]
            rows.append([problem, f'{a[0]}/{a[2]}', f'{b[0]}/{b[2]}', f'{a[1]}/{a[2]}', f'{b[1]}/{b[2]}'])
        parts += ["## Theo bài Validation", "", table(["Bài APPS", "Expected · 6 khối", "Expected · 24 khối", "Test bắt lỗi · 6 khối", "Test bắt lỗi · 24 khối"], rows), ""]
    parts += [conclusion, ""]
    if (folder / "evaluation-recovery.json").exists():
        parts += ["Seed 43 bị dừng khi khởi chạy Python, trước khi sinh test. Đã giữ lượt seed 42 và chạy nốt các seed còn thiếu với cùng adapter, prompt và ngân sách; không huấn luyện lại. Bản ghi nối lại nằm trong evaluation-recovery.json.", ""]
    for index, image in enumerate(figures, 1):
        caption = {"train-comparison.png": "Hai adapter SFT256 trên Train: 164 test, 41 code lỗi, seed sinh 42.",
                   "learning-curve.png": "Loss của hai adapter trên cùng thứ tự ví dụ Train; trung bình trượt 8 bước.",
                   "validation-comparison.png": "Validation: 60 test trên 15 code lỗi mỗi seed; cột là trung bình và thanh sai số là SD của ba seed sinh."}.get(image, title + ".") if folder.name == "apps-lora-input-first" else title + "."
        parts += [f'![{title}]({image})', "", f'Hình {index}. {caption}', ""]
    if folder.name == "apps-lora-input-first":
        parts += ['Qwen3.5-2B có kiến trúc xen kẽ linear attention và full attention; [cấu hình của model](https://huggingface.co/Qwen/Qwen3.5-2B) là cơ sở chọn các lớp mở rộng. Loss trên Train đo mức khớp dữ liệu học; kết luận chất lượng cần dựa vào đánh giá riêng.', ""]
    if folder.name == "apps-reward-ablation" and (folder / "sampling-metrics.csv").exists():
        parts += ["[Kết quả sinh lại với ba seed](SAMPLING.md).", ""]
    data_files = [p.name for p in sorted(folder.iterdir()) if p.suffix in (".csv", ".json") and p.name not in ("review.json", "ui-review.json")]
    if data_files:
        parts += ["## Dữ liệu", "", " · ".join(f'[{name}]({name})' for name in data_files), ""]
    parts += ["[Cấu hình và chỉ số đánh giá](../../docs/EXPERIMENTS.md).", ""]
    (folder / "README.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")


def write_sampling_notes(folder):
    folder = Path(folder)
    text = "# Reward: kết quả theo seed sinh\n\nCác checkpoint cố định được đánh giá với seed sinh 42/43/44, mỗi lượt 60 test trên 15 code Validation.\n\n"
    text += csv_table(folder / "sampling-metrics.csv")
    text += "\n\nMột seed train, ba seed sinh. Các số đếm theo seed được giữ để xem biến động lấy mẫu.\n\n"
    text += "![Kết quả theo seed](sampling-variability.png)\n\nHình 1. Kết quả sinh lại của các checkpoint.\n\n[Dữ liệu](sampling-metrics.json) · [Thí nghiệm reward](README.md).\n"
    (folder / "SAMPLING.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "results"
    for name in NOTES:
        folder = root / name
        if folder.is_dir():
            write_result_readme(folder)
    if (root / "apps-reward-ablation/sampling-metrics.csv").exists():
        write_sampling_notes(root / "apps-reward-ablation")
