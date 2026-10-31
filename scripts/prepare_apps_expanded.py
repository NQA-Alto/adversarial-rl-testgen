import hashlib
import json
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

from apps_core import ASSETS, DATA, SOURCE, REVISION, all_inputs, canonical, digest, model_prompt, read, write
from apps_profiles import integer as I, sequence as L, text as T
from prepare_apps import mutations, references
from restricted_python import RestrictedFunction


def E(kind, values):
    return {'type': kind, 'enum': values}


PROFILES = {
    2722: ('Bỏ phần neo của URL', [E('str', ['https://a.vn', 'https://a.vn#b', 'https://a.vn?a=1#x', 'https://b.vn#', 'https://b.vn?q=1'])]),
    2796: ('Tên có bắt đầu bằng R hoặc r?', [E('str', ['Adam', 'Ringo', 'robert', 'Anna', 'Dora'])]),
    2823: ('Các giá trị xuất hiện lặp theo thứ tự lần lặp đầu tiên', [L(maximum=5)]),
    2827: ('Viết chữ cho chữ số từ 0 đến 9', [I(0, 9)]),
    2876: ('Phần tử có nằm trong danh sách?', [L(maximum=3), I(-2, 3)]),
    2905: ('Tạo biệt danh ba hoặc bốn ký tự', [E('str', ['Sam', 'Jimmy', 'Samantha', 'Kayne', 'Robert', 'Anna', 'An'])]),
    3031: ('Chuỗi có thứ tự bảng chữ cái?', [T('abc', 1, 5)]),
    3045: ('Chọn thang máy gần nhất, ưu tiên bên phải khi hòa', [I(0, 2), I(0, 2), I(0, 2)]),
    3054: ('Dãy tổng cộng dồn có dấu', [I(-6, 6)]),
    3061: ('Số lần xuất hiện của phần tử phổ biến nhất', [L(maximum=5)]),
    3075: ('Đếm nghịch thế trong danh sách', [L(maximum=4)]),
    3080: ('Tên đầy đủ và tên rút gọn hai ký tự', [E('str', ['An', 'Anna', 'Robert', 'Nam', 'A'])]),
    3094: ('Tổng sau khi bỏ đúng một phần tử nhỏ nhất và lớn nhất', [L(maximum=5)]),
    3126: ('Có thể sắp xếp lại chuỗi thành palindrome?', [T('abc', 4, 5)]),
    3136: ('Chọn đồ uống theo độ tuổi', [I(0, 25)]),
    3144: ('Đếm số đôi găng tay cùng màu', [L(('red', 'green', 'blue'), 0, 5)]),
    3181: ('Nhân vật còn sống khi máu lớn hơn không', [I(-10, 10)]),
    3192: ('Phân loại số lượng chó Dalmatian', [I(1, 101)]),
    3237: ('Số chẵn hay lẻ', [I(-8, 8)]),
    3324: ('Tính tiền hotdog theo ba mức số lượng', [I(1, 15)]),
    3369: ('Vị trí mới sau hai lần số bước xúc xắc', [I(0, 8), I(1, 6)]),
    3381: ('Đổi tầng kiểu Mỹ sang tầng kiểu châu Âu', [E('int', [-3, -2, -1, 0, 1, 2, 3, 5, 12, 14, 15, 16])]),
    3418: ('Đảo thứ tự danh sách', [L(maximum=4)]),
    3457: ('Tính điểm cuối từ điểm thi và số dự án', [E('int', [0, 50, 51, 75, 76, 90, 91, 100]), I(0, 12)]),
    3464: ('Kiểm tra năm nhuận', [E('int', [1, 4, 100, 400, 1900, 1996, 2000, 2004, 2100])]),
    3482: ('Liệt kê số có bit đầu và cuối bằng một', [I(1, 12)]),
    3506: ('Vị trí nguyên âm tính từ một', [T('aEyB', 0, 4)]),
    3510: ('Số hạt đỏ giữa các hạt xanh', [I(0, 10)]),
    3519: ('Hai danh sách có phần tử chung?', [L(maximum=3), L(maximum=3)]),
    3522: ('Khoảng cách lớn nhất giữa hai phần tử kề sau sắp xếp', [L(minimum=3, maximum=4)]),
    3547: ('Giữ phần tử có số lần xuất hiện chẵn', [L(maximum=5)]),
    3564: ('Chuỗi nhị phân xen kẽ bắt đầu bằng một', [I(1, 12)]),
    3568: ('Xe có vượt qua tối đa mười lăm chỗ xóc?', [E('str', ['_', 'n', 'n'*14, 'n'*15, 'n'*16, 'n'*17, '_n_'*15, '_n_'*16])]),
    3572: ('Có cần mời thêm phụ nữ?', [L((-1, 1), 2, 6)]),
    3578: ('Số trang giấy cần sao chép', [I(-3, 5), I(-3, 5)]),
    3603: ('Hai bông hoa có số cánh khác tính chẵn lẻ?', [I(0, 6), I(0, 6)]),
    3617: ('Danh sách có tổng không và phần tử đối dấu?', [L((-2, -1, 0, 1, 2), 0, 4)]),
    3693: ('Đưa số về giá trị không dương', [I(-8, 8)]),
    3709: ('Nhân số chẵn với tám và số lẻ với chín', [I(-6, 8)]),
    3721: ('Đếm lần đổi dấu giữa các phần tử liên tiếp', [L((-2, -1, 1, 2), 0, 4)]),
    3724: ('Đạn có đủ cho hai viên mỗi rồng?', [I(0, 12), I(0, 6)]),
    3728: ('Phân loại danh sách rỗng, một hay nhiều phần tử', [L(maximum=4)]),
    3736: ('Phần tử nhỏ nhất trong danh sách không rỗng', [L(minimum=1, maximum=4)]),
    3779: ('Đổi giờ phút giây thành mili giây', [I(0, 3), E('int', [0, 1, 2, 58, 59]), E('int', [0, 1, 2, 58, 59])]),
    3795: ('Lượng máu còn lại không âm sau sát thương', [I(0, 10), I(0, 10)]),
    3802: ('Thông điệp động viên theo số vòng lắc', [I(0, 15)]),
    3807: ('Số phần tử nhỏ hơn ở bên phải mỗi vị trí', [L(maximum=4)]),
}


def main():
    from transformers import AutoTokenizer
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    target = ASSETS / 'data/apps-introductory-expanded-v4'
    if target.exists():
        raise FileExistsError('Frozen version already exists')
    base_manifest = read(DATA / 'manifest.json')
    base_problems = read(DATA / 'problems.json')
    reserved = [p for p in base_problems if p['split'] != 'train']
    reserved_urls = {p['source_url'].rstrip('/') for p in reserved}
    reserved_specs = {canonical(p['specification']) for p in reserved}
    old_ids = {p['source_id'] for p in base_problems}
    rows = [json.loads(line) for line in (SOURCE / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(ASSETS / 'models/Qwen3.5-2B', local_files_only=True)
    cases = [json.loads(line) for line in (DATA / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted, excluded, checks = [], [], []
    for identity, (title, fields) in PROFILES.items():
        try:
            row = rows[identity]
            assert identity not in old_ids
            assert row['difficulty'] == 'introductory'
            if row['url'].rstrip('/') in reserved_urls or canonical(row['question']) in reserved_specs:
                raise ValueError('Overlap with reserved Validation/Test')
            io, refs = references(row)
            arguments = refs[0]['arguments']
            if len(fields) != len(arguments):
                raise ValueError('Profile arity mismatch')
            problem = {'problem_id': f'apps-train-{identity}', 'source_id': identity, 'source_split': 'train', 'source_revision': REVISION, 'source_url': row['url'], 'source_entry_point': io['fn_name'], 'title': title, 'specification': row['question'], 'split': 'train', 'domain': dict(zip(arguments, fields, strict=True)), 'arguments': arguments, 'constraints': [], 'reference': refs[0]['code'], 'references': refs, 'source_record_sha256': hashlib.sha256(canonical(row).encode()).hexdigest(), 'source_tests': io, 'scope': 'Explicit finite valid subdomain; domain chosen by reading original specification, no model-score selection'}
            inputs = list(all_inputs(problem))
            if not 1 <= len(inputs) <= 10000:
                raise ValueError('Finite-domain budget')
            runners = [RestrictedFunction(ref['code'], ref['arguments']) for ref in refs]
            gold = []
            for value in inputs:
                outputs = [fn(**dict(zip(ref['arguments'], list(value.values()), strict=True))) for fn, ref in zip(runners, refs, strict=True)]
                if canonical(outputs[0]) != canonical(outputs[1]):
                    raise ValueError('References disagree in declared domain')
                gold.append(outputs[0])
            vectors = {canonical(gold)}
            retained = []
            reasons = Counter()
            for code, edit in mutations(problem['reference']):
                try:
                    fn = RestrictedFunction(code, arguments)
                    output = [fn(**value) for value in inputs]
                    vector = canonical(output)
                    if vector in vectors:
                        reasons['equivalent_or_duplicate'] += 1
                        continue
                    same = [canonical(a) == canonical(b) for a,b in zip(gold, output, strict=True)]
                    if not any(same):
                        reasons['fails_every_input'] += 1
                        continue
                    case = {k:v for k,v in problem.items() if k not in ('references','source_tests')}
                    case.update(case_id=problem['problem_id']+f'/m{len(retained)+1}', buggy_code=code, bug_origin='single_AST_mutation_of_public_reference', mutation=edit)
                    prompt_text = tokenizer.apply_chat_template(model_prompt(case, 'apps-call-v2'), tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
                    assert len(prompt) > 20
                    if len(prompt) > 1536:
                        raise ValueError('Prompt exceeds frozen 1536-token budget')
                    retained.append(case)
                    vectors.add(vector)
                    witness = next(i for i, flag in enumerate(same) if not flag)
                    checks.append({'case_id': case['case_id'], 'finite_inputs': len(inputs), 'different_inputs': len(same)-sum(same), 'same_inputs': sum(same), 'witness': {'input': inputs[witness], 'expected': gold[witness], 'buggy_output': output[witness]}, 'prompt_tokens_2b': len(prompt)})
                    if len(retained) == 3:
                        break
                except Exception as error:
                    reasons[str(error)] += 1
            if not retained:
                raise ValueError('No supported nontrivial mutant: '+canonical(dict(reasons)))
            problem['audit'] = {'references': 2, 'source_tests_per_reference': len(io['inputs']), 'finite_domain_inputs': len(inputs), 'mutants': len(retained), 'rejected_mutants': dict(reasons)}
            cases.extend(retained)
            accepted.append(problem)
            print('ACCEPT', identity, len(inputs), len(retained), flush=True)
        except Exception as error:
            excluded.append({'source_id': identity, 'reason': str(error)})
            print('EXCLUDE', identity, str(error), flush=True)
    target.mkdir()
    (target / 'train.jsonl').write_text(''.join(canonical(c)+'\n' for c in cases), encoding='utf-8', newline='\n')
    for split in ('validation', 'test'):
        shutil.copy2(DATA / (split+'.jsonl'), target / (split+'.jsonl'))
    manifest = {'dataset': 'APPS Introductory bounded call-based expanded Train', 'source': 'codeparrot/apps', 'source_revision': REVISION, 'base_dataset_id': base_manifest['dataset_id'], 'base_manifest_sha256': digest(DATA / 'manifest.json'), 'selection': '47 additional prespecified source profiles reviewed against original specification; only Train expanded; two source references, full finite-domain agreement, distinct nontrivial AST mutants; 1536 token prompt limit; no selection by model scores', 'official_test_downloaded': False, 'official_test_evaluated': False, 'split_policy': base_manifest['split_policy'], 'splits': {}, 'added_problems': len(accepted), 'excluded': excluded, 'original_validation_test_bytes_preserved': True}
    for split in ('train','validation','test'):
        path = target / (split+'.jsonl')
        if split == 'train':
            counts = {'cases':len(cases), 'problems':len({c['problem_id'] for c in cases})}
        else:
            counts = {k:base_manifest['splits'][split][k] for k in ('cases','problems')}
            assert digest(path) == base_manifest['splits'][split]['sha256']
        manifest['splits'][split] = dict(counts, sha256=digest(path))
    manifest['dataset_id'] = 'apps-intro-expanded-v4-' + hashlib.sha256(canonical(manifest).encode()).hexdigest()[:12]
    write(target / 'manifest.json', manifest)
    write(target / 'problems.json', base_problems+accepted)
    write(target / 'expansion-audit.json', {'cases':checks, 'new_finite_case_checks':sum(r['finite_inputs'] for r in checks), 'excluded':excluded})
    print('EXPANDED', canonical(manifest), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
