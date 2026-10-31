import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
sys.path.insert(0, str(PROJECT / "scripts"))
from apps_core import canonical, digest, load_cases, read, score, write
from apps_adversary import rows
from audit_apps_run import audit


def main(root):
    result, plan = read(root / "result.json"), read(root / "plan.json")
    if result["status"] != "completed":
        raise ValueError("Study is not complete")
    for path, expected in plan["guards"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Protected evaluation changed")
    output = PROJECT / "results/apps-balanced-adversarial"
    output.mkdir(exist_ok=True)
    comparisons, exposure, sampling, audited_runs = [], [], [], 0
    validation_cases = {case['case_id']: case for case in load_cases('validation')[0]}
    baseline_study = Path(read(ASSETS/'evidence/APPS_EXPANDED_STUDY.json')['path'])
    baseline_prompts = {}
    for record in read(baseline_study/'result.json')['validation_runs']['sft']:
        folder = Path(record['path'])
        report = read(folder/'result.json')
        seed = report.get('seed', report.get('protocol', {}).get('seed'))
        if not isinstance(seed, int):
            raise ValueError('Missing recorded generation seed')
        baseline_prompts[seed] = {row['case_id']:(row['prompt_sha256'], row.get('case_seed',
            (seed + int(hashlib.sha256(row['case_id'].encode()).hexdigest()[:8], 16)) % 2**32))
            for row in rows(folder/'predictions.jsonl')}
    groups_total = signal_total = tests_total = 0
    per_problem = []
    for seed in result["seeds"]:
        previous_curriculum = None
        for stage in seed["rounds"]:
            pool = Path(stage['pool'])
            pool_report = read(pool/'result.json')
            current_curriculum = {case['case_id']: case for case in rows(root/f"seed-{seed['seed']}"/f"round-{stage['round']}-adaptive-curriculum.jsonl")}
            eligible = {identity for suites in read(pool/'policy-suites.json').values() for identity in suites}
            changed = retained = 0
            for identity, case in current_curriculum.items():
                if previous_curriculum is None:
                    changed += case.get('curriculum_source') == 'local_adversary'
                else:
                    differs = case['buggy_code'] != previous_curriculum[identity]['buggy_code']
                    if differs and identity not in eligible:
                        raise ValueError('Refresh replaced a slot not solved by the current policy')
                    changed += differs
                    retained += not differs and case.get('curriculum_source') == 'local_adversary'
            sampling.append({'training_seed':seed['seed'],'round':stage['round'],
                             'sampled_candidates':pool_report['sampled_candidates'],
                             'accepted_candidates':pool_report['accepted_candidates'],
                             'eligible_slots':len(eligible),'replaced_slots':changed,'retained_adversarial_slots':retained})
            previous_curriculum = current_curriculum
            control = read(Path(stage["arms"]["control"]["training"]["path"]) / "schedule.json")
            adaptive = read(Path(stage["arms"]["adaptive"]["training"]["path"]) / "schedule.json")
            if control != adaptive:
                raise ValueError("Arms received different case-ID schedules")
            for arm in ("control", "adaptive"):
                folder = Path(stage["arms"][arm]["training"]["path"])
                audit(folder)
                events, gradients = rows(folder / "rollouts.jsonl"), read(folder / "gradients.json")
                focus = set(control["focus_case_ids"])
                focused = [event for event in events if event["case_id"] in focus]
                signal = sum(event["policy_signal"] and gradients[event["optimizer_step_before"]]["norm"] > 0 for event in focused)
                tested = sum(len(event["samples"]) for event in focused)
                if len(focused) != 32 or tested != 128:
                    raise ValueError("Unexpected realized exposure")
                exposure.append({"training_seed": seed["seed"], "round": stage["round"], "arm": arm,
                                 "focus_slots": stage["focus_slots"], "focus_groups": len(focused),
                                 "focus_tests": tested, "focus_groups_with_update": signal})
                if arm == "adaptive":
                    groups_total += len(focused)
                    signal_total += signal
                    tests_total += tested
        for arm in ("control", "adaptive"):
            for record in seed["validation"][arm]["runs"]:
                folder = Path(record["path"])
                audit(folder)
                audited_runs += 1
                report = read(folder / "result.json")
                signatures = {row['case_id']:(row['prompt_sha256'],row['case_seed']) for row in rows(folder/'predictions.jsonl')}
                if signatures != baseline_prompts[report['seed']]:
                    raise ValueError('Validation prompt or sampling seeds differ from SFT baseline')
                count = {"variant": arm, "training_seed": seed["seed"], "generation_seed": report["seed"],
                         **{key: report["metrics"][key] for key in ("valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")},
                         "input_attack_tests": 0, "missed_due_to_expected": 0}
                grouped = {}
                for row in rows(folder / "predictions.jsonl"):
                    bucket = grouped.setdefault(row["source_id"], {"codes": 0, "detected": 0, "valid": 0, "correct": 0, "hits": 0})
                    bucket["codes"] += 1
                    bucket["detected"] += any(item["bug_detected"] for item in row["samples"])
                    for sample in row["samples"]:
                        bucket["valid"] += sample["valid_format"]
                        bucket["correct"] += sample["correct_expected"]
                        bucket["hits"] += sample["bug_detected"]
                        if sample['valid_format']:
                            recomputed = score(sample['raw_output'], sample['terminated'], validation_cases[row['case_id']])
                            corrected = canonical({'input': recomputed['test']['input'], 'expected': recomputed['gold_output']})
                            input_hit = score(corrected, True, validation_cases[row['case_id']])['bug_detected']
                            count['input_attack_tests'] += input_hit
                            count['missed_due_to_expected'] += input_hit and not sample['correct_expected']
                for identity, values in grouped.items():
                    per_problem.append({"variant": arm, "training_seed": seed["seed"], "generation_seed": report["seed"], "source_id": identity, **values})
                comparisons.append(count)
    seed_count = len(result['seeds'])
    if tests_total != 256 * seed_count or audited_runs != 6 * seed_count:
        raise ValueError("Incomplete experiment")
    for name, values in (("runs.csv", comparisons), ("exposure.csv", exposure), ("per-problem.csv", per_problem), ('sampling.csv',sampling)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    public = {"dataset_id": plan["dataset_id"], "model": "Qwen3.5-2B BF16", "parent": "SFT 840",
              "parent_adapter_sha256": plan["parent_adapter_sha256"], "training_seeds": [seed['seed'] for seed in result['seeds']],
              "generation_seeds": plan["generation_seeds"], "rounds": 2, "steps_per_round_per_arm": 64,
              "learning_rate": plan["learning_rate"], "reward_profile": plan["reward_profile"],
              "tests_per_case": 4, "validation_cases": 15, "validation_problems": 5,
              "sft": result["sft_validation"], "summary": result["summary"],
              "adversarial_groups": groups_total, "adversarial_tests": tests_total,
              "adversarial_groups_with_update": signal_total,
              "adaptive_passes_selection_gate": result["adaptive_passes_selection_gate"],
              "old_test_unchanged": True, "new_model_test_evaluation": False,
              "sampling_schedule_verified": True, "validation_prompt_and_seed_match": True, "source_hashes": plan["source_hashes"],
              "confirmation_rule": plan.get('confirmation_rule'),
              "confirmation_skipped_reason": result.get('confirmation_skipped_reason'),
              "refresh_context": "Current curriculum; retain unresolved code, replace solved slots only"}
    reused = bool(plan.get('reused_round1_study'))
    if reused:
        certificate = read(root/'round1-reuse-review.json')
        if certificate['status'] != 'passed' or certificate['checks']['after_sha256'] != plan['source_hashes']['apps_model_runner.py']:
            raise ValueError('Missing round1 equivalence certificate')
    public.update(refresh_verified=True,round1_reused=reused,round1_training_functions_unchanged=True,sampling=sampling)
    write(output / "metrics.json", public)
    model_keys = ("sft", "control", "adaptive")
    names = ("SFT 840", "GRPO cố định", "GRPO đối kháng")
    mean = lambda variant, metric: (public["sft"] if variant == "sft" else public["summary"][variant])[metric]["mean"]
    lines = ["# Lấy mẫu cân bằng cho GRPO đối kháng", "",
             "Hai nhánh cùng bắt đầu từ SFT 840 trên tập Train 58 bài/145 code. Mỗi nhánh học hai vòng 64 bước, learning rate 5e-6, reward chia đều. Seed train đã chạy: " + ', '.join(str(seed['seed']) for seed in result['seeds']) + ". Mỗi checkpoint cuối được chấm trên ba seed sinh 42, 43, 44.", "",
             "Một nửa nhóm học lấy từ các vị trí có code đối kháng, một nửa từ phần Train còn lại. Nhánh cố định dùng cùng lịch case ID nhưng giữ code AST ban đầu. Code đối kháng được làm mới sau vòng đầu; model sinh code giữ nguyên trọng số.", "",
             "## Validation", "", "| Phiên bản | Expected đúng /60 | Test bắt lỗi /60 | Code phát hiện /15 |", "|---|---:|---:|---:|"]
    for variant, name in zip(model_keys, names):
        cells = []
        for metric in ("correct_expected", "bug_detecting_tests", "cases_detected"):
            number = mean(variant, metric)
            text = f"{number:.2f}"
            if variant != "sft" and number > mean("sft", metric) + 1e-9:
                text = "**" + text + "**"
            cells.append(text)
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += ["", f"SFT là một checkpoint cố định với ba seed sinh. Mỗi nhánh GRPO có {seed_count} lượt train và {3 * seed_count} lượt đánh giá. Seed sinh không phải seed huấn luyện. Phần in đậm cao hơn SFT, không biểu thị kiểm định ý nghĩa thống kê.", "",
              "![Kết quả Validation](validation.png)", "", "Hình 1. Số test có expected đúng và số code phát hiện trên Validation. Chấm nhỏ là từng lượt đánh giá GRPO, cột là trung bình.", "",
              "## Lượng học trên code đối kháng", "", "| Seed train | Vòng | Vị trí đối kháng | Nhóm học | Test sinh | Nhóm có cập nhật |", "|---|---:|---:|---:|---:|---:|"]
    for row in exposure:
        if row["arm"] == "adaptive":
            lines.append(f'| {row["training_seed"]} | {row["round"]} | {row["focus_slots"]} | {row["focus_groups"]} | {row["focus_tests"]} | {row["focus_groups_with_update"]} |')
    a, c = public["summary"]["adaptive"], public["summary"]["control"]
    delta_codes = a["cases_detected"]["mean"] - c["cases_detected"]["mean"]
    delta_expected = a["correct_expected"]["mean"] - c["correct_expected"]["mean"]
    lines += ["", f"Nhánh đối kháng có {groups_total} nhóm thực sự học trên code mới, tương ứng {tests_total} test; {signal_total} nhóm vừa có chênh lệch reward vừa có gradient khác 0.", "",
              "## Nhận xét", "", f"So với nhánh cố định cùng điều kiện, đối kháng thay đổi {delta_codes:+.2f} code phát hiện/15 và {delta_expected:+.2f} expected đúng/60. Điều kiện chọn ứng viên từ Validation " + ("đã đạt." if result["adaptive_passes_selection_gate"] else "chưa đạt; giữ SFT làm bản sử dụng."), "",
              "Vòng hai kiểm tra policy trên code thực sự đang học. Code chưa bị bắt được giữ lại; chỉ vị trí đã được policy giải quyết mới tìm code thay thế. Một lượt trước đó đã dừng ở bước lấy mẫu khi phát hiện nó còn kiểm tra trên AST gốc; checkpoint vòng một được giữ và chấm lại vì phần huấn luyện không đổi.", "",
              "Seed 43 chỉ chạy xác nhận nếu pilot seed 42 đạt điều kiện cải thiện đã ghi trong protocol. " + ("Pilot chưa có lợi thế đạt điều kiện nên dừng ở một seed train." if result.get('confirmation_skipped_reason') else "Pilot đủ điều kiện chạy xác nhận."), "",
              "Lượt học này diễn ra sau sự kiện Test của phiên bản trước. Test cũ không chạy lại và không được dùng để báo kết quả checkpoint mới. Các số ở đây chỉ là Validation đã được dùng nhiều lần trong quá trình phát triển.", "",
              "Lấy mẫu, learning rate và ngân sách khác đợt trước, nên không quy toàn bộ chênh lệch với đợt trước cho riêng lịch cân bằng. Đối chứng trong đợt này chỉ khác code tại vị trí đối kháng.", "",
              "Mỗi vòng lưu curriculum, lịch học, rollout, gradient, hash adapter và mã nguồn. `runs.csv`, `exposure.csv` và `per-problem.csv` giữ các số đếm để kiểm tra bảng tổng hợp.", ""]
    lines += ['## Expected và khả năng chọn input', '',
              '| Nhánh | Input có thể bắt lỗi /60 | Cơ hội mất do expected sai /60 |', '|---|---:|---:|']
    for arm, label in (('control', 'Cố định'), ('adaptive', 'Đối kháng')):
        values = [row for row in comparisons if row['variant'] == arm]
        lines.append(f"| {label} | {statistics.mean(row['input_attack_tests'] for row in values):.2f} | {statistics.mean(row['missed_due_to_expected'] for row in values):.2f} |")
    lines += ['', 'Bảng này chỉ phân tích input: expected được thay bằng output của reference để đo cơ hội bắt lỗi đã mất. Đây không phải kết quả sinh test hoàn chỉnh của model và không thay số liệu Attack rate ở bảng chính.', '']
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    fig.patch.set_facecolor("#fcfcfa")
    for axis, metric, maximum, title in zip(axes, ("correct_expected", "cases_detected"), (60, 15),
                                           ("Expected đúng", "Code lỗi được phát hiện")):
        numbers = [mean(variant, metric) for variant in model_keys]
        label_height = max(numbers + [row[metric] for row in comparisons]) + maximum * 0.06
        axis.set_facecolor("#fcfcfa")
        axis.bar(range(3), numbers, color=("#a5a99d", "#8a9f8b", "#466e54"), width=0.6, zorder=2)
        for index, value in enumerate(numbers):
            axis.text(index, label_height, f"{value:.2f}", ha="center")
        for index, arm in enumerate(("control", "adaptive"), start=1):
            values = [row[metric] for row in comparisons if row["variant"] == arm]
            offsets = [-0.10 + 0.20 * i / max(len(values) - 1, 1) for i in range(len(values))]
            axis.scatter([index + offset for offset in offsets], values, s=18, color="#263f2e", zorder=3)
        axis.set_xticks(range(3), names)
        axis.set_ylim(0, maximum)
        axis.set_yticks(range(0, maximum + 1, 10 if maximum == 60 else 3))
        axis.set_ylabel("Số test /60" if maximum == 60 else "Số code /15")
        axis.set_title(title, loc="left", pad=14)
        axis.grid(axis="y", alpha=0.17, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "validation.png", dpi=160)
    plt.close(fig)
    write(root / "publication-review.json", {"status": "passed", "training_runs_audited": 4 * seed_count,
          "validation_runs_audited": audited_runs, "exact_schedules_verified": 2 * seed_count,
          "adversarial_groups": groups_total, "adversarial_tests": tests_total,
          "groups_with_verified_update": signal_total, "old_test_unchanged": True})
    print("Balanced results ready:", output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
