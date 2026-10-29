import base64
import csv
import hashlib
import html
import json
import re
import statistics
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "docs/lesson"
OUTPUT = PROJECT / "docs/BAI_GIANG_DO_AN_TTNT.html"
ASSETS = Path("D:/DA/TTNT_De19_Assets")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def image(path, alt):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return '<img src="data:image/png;base64,' + encoded + '" alt="' + html.escape(alt, quote=True) + '">'


def table(headers, records, highlight=False):
    maxima = {i:max(float(row[i]) for row in records) for i in range(1,len(headers))} if highlight else {}
    def cell(value,i):
        text = html.escape(str(value))
        return '<td>' + ('<strong>'+text+'</strong>' if i in maxima and float(value)==maxima[i] else text) + '</td>'
    result = '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + html.escape(str(value)) + '</th>' for value in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(cell(value,i) for i,value in enumerate(row)) + '</tr>' for row in records) + '</tbody></table></div>'
    return result + ('<p class="count">Số in đậm là giá trị lớn nhất mỗi cột, kể cả khi hòa.</p>' if highlight else '')


def code(text):
    return '<pre><code>' + html.escape(text.strip()) + '</code></pre>'


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return f'{value:.3f}'.replace('.', ',')


PIPELINE = '''<figure><svg viewBox="0 0 960 365" role="img" aria-labelledby="pipeline-title"><title id="pipeline-title">Luồng sinh test và nhánh cập nhật adapter khi train</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#617065"/></marker><marker id="green-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#35634c"/></marker></defs>
<g fill="#f7f8f3" stroke="#cbd2c9"><rect x="20" y="35" width="205" height="100" rx="6"/><rect x="270" y="35" width="205" height="100" rx="6"/><rect x="520" y="35" width="205" height="100" rx="6"/><rect x="770" y="35" width="170" height="100" rx="6"/><rect x="770" y="185" width="170" height="73" rx="6"/></g><rect x="270" y="207" width="205" height="73" rx="6" fill="#e8f2e9" stroke="#85a58d"/>
<g text-anchor="middle" fill="#293c30" font-family="Segoe UI,Arial,sans-serif"><text x="122" y="76" font-size="20">Đề + code</text><text x="122" y="105" font-size="15">Input cho policy</text><text x="373" y="76" font-size="20">Qwen + LoRA</text><text x="373" y="105" font-size="15">Sinh bốn test</text><text x="623" y="76" font-size="20">Input / expected</text><text x="623" y="105" font-size="15">Đầu ra do model sinh</text><text x="855" y="76" font-size="20">Kiểm chứng</text><text x="855" y="105" font-size="14">Đúng? Bắt lỗi?</text><text x="855" y="215" font-size="16">Reference</text><text x="855" y="240" font-size="14">Chỉ phía bộ chấm</text><text x="373" y="238" font-size="18">GRPO cập nhật</text><text x="373" y="263" font-size="15">Chỉ adapter LoRA</text></g>
<g stroke="#617065" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M225 85H263"/><path d="M475 85H513"/><path d="M725 85H763"/><path d="M855 185V143"/></g><g stroke="#35634c" stroke-width="2.5" fill="none" marker-end="url(#green-arrow)"><path d="M940 86H953V309H373V285"/><path d="M373 207V143"/></g><text x="617" y="300" font-family="Segoe UI,Arial,sans-serif" font-size="17" fill="#35634c">Reward · chỉ đi nhánh này khi train</text><text x="20" y="351" font-family="Segoe UI,Arial,sans-serif" font-size="16" fill="#59675d">Demo / đánh giá: sinh và chấm; không cập nhật trọng số.</text></svg><figcaption>Luồng cơ bản. Vòng adversary làm mới code Train được giải thích riêng ở mục 12.</figcaption></figure>'''

LORA = '''<figure><svg viewBox="0 0 850 210" role="img" aria-labelledby="lora-title"><title id="lora-title">Model nền đóng băng cộng phần điều chỉnh LoRA được học</title><rect x="15" y="35" width="220" height="118" rx="6" fill="#f0f1ed" stroke="#c8ccc1"/><rect x="325" y="35" width="200" height="118" rx="6" fill="#e7f1e8" stroke="#8dad94"/><rect x="615" y="35" width="220" height="118" rx="6" fill="#f7f8f3" stroke="#c8ccc1"/><g font-family="Segoe UI,Arial,sans-serif" text-anchor="middle" fill="#2c4031"><text x="125" y="77" font-size="21">Qwen nền</text><text x="125" y="109" font-size="17">Giữ nguyên W</text><text x="125" y="135" font-size="15">Hàng tỷ tham số</text><text x="425" y="77" font-size="21">LoRA</text><text x="425" y="109" font-size="17">Học A và B</text><text x="425" y="135" font-size="15">208.896 tham số</text><text x="725" y="77" font-size="21">Model sử dụng</text><text x="725" y="109" font-size="17">Nền + adapter</text><text x="725" y="135" font-size="15">Kết quả có thể thay đổi</text><text x="280" y="108" font-size="37">+</text><text x="570" y="108" font-size="37">=</text></g><text x="425" y="190" text-anchor="middle" font-size="16" fill="#59675d" font-family="Segoe UI,Arial,sans-serif">Adapter nhỏ vẫn cần đúng model nền để nạp và suy luận.</text></svg><figcaption>LoRA trong project: nền đóng băng, phần điều chỉnh được SFT hoặc GRPO cập nhật.</figcaption></figure>'''

CSS = r'''
:root{--paper:#fff;--bg:#f5f3ed;--ink:#29372f;--muted:#627065;--green:#34624c;--line:#dfe3d9;--text:18px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--bg);color:var(--ink);font:var(--text)/1.8 "Segoe UI",Arial,sans-serif}a{color:#285c49;text-underline-offset:3px}a:hover{color:#174232}button,input,select{font:inherit}button{cursor:pointer;border:1px solid #c8d2c8;border-radius:4px;background:#fff;color:var(--ink);padding:7px 12px}button:hover{background:#edf3eb}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #a0bca4;outline-offset:3px}input[type=number]{width:120px;padding:7px;border:1px solid #c7cfc3;border-radius:4px}select{max-width:100%;padding:6px;border:1px solid #c7cfc3;background:#fff;color:var(--ink)}.shell{display:grid;grid-template-columns:260px minmax(0,960px);gap:38px;max-width:1330px;padding:28px 28px 60px;margin:auto}.sidebar{position:sticky;top:22px;align-self:start;max-height:calc(100vh - 44px);overflow:auto;font-size:14px;line-height:1.5;padding-right:10px}.sidebar small{display:block;color:var(--muted);margin-bottom:14px}.sidebar summary{font-weight:650;cursor:pointer;margin:0 0 14px}.sidebar a{display:block;padding:7px 9px;text-decoration:none;border-left:2px solid transparent}.sidebar a:hover,.sidebar a.active{background:#eaf0e6;border-left-color:var(--green)}main{min-width:0}.cover,section{background:var(--paper);border:1px solid var(--line);padding:34px 40px;margin:0 0 24px;border-radius:3px}.cover{padding-top:45px;border-top:4px solid var(--green)}.cover h1{font:600 43px/1.2 Georgia,"Times New Roman",serif;letter-spacing:-.5px;margin:12px 0 17px}.subtitle{font-size:18px;color:var(--muted);line-height:1.6}.meta{font-size:14px;color:var(--muted);padding-top:12px}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}.read-routes{font-size:16px;border-top:1px solid var(--line);padding-top:12px}.kicker{text-transform:uppercase;letter-spacing:.09em;font-size:12px;font-weight:700;color:var(--green);margin:0 0 14px}h2{font:600 29px/1.35 Georgia,"Times New Roman",serif;margin:0 0 22px;scroll-margin-top:24px}h3{font-size:20px;line-height:1.45;margin:29px 0 10px}p{margin:0 0 16px}strong{font-weight:650}ol,ul{padding-left:25px;margin:12px 0 20px}li{padding-left:3px;margin:6px 0}.key,.note{padding:17px 20px;background:#edf4eb;border-left:3px solid #799b7a;margin:22px 0}.note{font-size:15px;background:#f7f3e8;border-left-color:#b4a27b}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{border:1px solid var(--line);padding:20px}.card h3{margin-top:0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.65;margin:18px 0 22px}th,td{border-bottom:1px solid var(--line);padding:11px 12px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#f1f4ed;font-weight:650}tr:last-child td{border-bottom:1px solid #c9d0c4}code{font: .86em/1.65 Consolas,"Courier New",monospace;color:#294537;background:#f0f3ed;padding:2px 4px;border-radius:2px;overflow-wrap:anywhere}pre{margin:15px 0 20px;padding:18px;background:#f2f4ef;border:1px solid #e0e4da;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65}pre code{padding:0;background:none;font-size:14px;color:#263d2c}details.question,details.deep{border-top:1px solid var(--line);padding:15px 0}details:last-child{border-bottom:1px solid var(--line)}summary{cursor:pointer;font-weight:600;line-height:1.6}details p,details ol{margin-top:15px}.deep{font-size:16px}.equation{font:18px/2 Georgia,"Times New Roman",serif;padding:16px 20px;background:#f6f7f1;border:1px solid var(--line);margin:18px 0;overflow-x:auto}.formula-inline{font-family:Georgia,"Times New Roman",serif}figure{margin:27px 0 32px}figure img,figure svg{width:100%;height:auto;display:block}figure svg{background:white;border:1px solid var(--line)}figcaption{font-size:14px;line-height:1.6;color:var(--muted);margin-top:10px}figure.screen{border:1px solid var(--line);padding:10px}figure.screen img{border:1px solid #e8e8df}.source{font-size:14px;color:var(--muted)}.widget{border:1px solid #bbcebc;padding:23px;background:#fafcf8;margin:24px 0}.widget h3{margin-top:0}.widget p{font-size:15px}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:14px;margin:18px 0}.controls label{font-size:15px;display:flex;align-items:center;gap:7px}.widget-result{padding:15px;background:#eaf2e7;border:1px solid #d8e5d4;line-height:1.7;font-size:16px}.widget table select{font-size:14px;width:100%}.widget table{margin:14px 0}.bar-row{display:flex;align-items:center;gap:10px;font-size:14px;padding:7px 0}.bar-row>span{width:44px}.bar-track{height:15px;flex:1;background:#e7ece3;position:relative;border-radius:2px}.bar-track:before{content:"";position:absolute;left:50%;top:-3px;height:21px;width:1px;background:#89978a}.bar-fill{position:absolute;height:15px;background:#628d6c}.bar-fill.negative{background:#c29273}.bar-row b{width:68px;text-align:right;font-weight:500}.speech{border-left:3px solid #a9bba8;padding-left:22px}.speech p{font-size:17px}.foot{font-size:14px;color:var(--muted);text-align:center;padding:10px}.no-js{padding:10px;background:#fff6df} .count{color:var(--muted);font-size:13px}
@media(max-width:950px){.shell{grid-template-columns:1fr;padding:15px;gap:18px}.sidebar{position:static;max-height:none;padding:14px 18px;background:#fff;border:1px solid var(--line)}.sidebar a{padding:8px}.sidebar small{margin-bottom:8px}.cover,section{padding:27px 24px}.cover h1{font-size:36px}.two{grid-template-columns:1fr}.cover{padding-top:30px}}
@media(max-width:550px){body{font-size:17px}.shell{padding:9px}.cover,section{padding:24px 18px}.cover h1{font-size:31px}h2{font-size:25px}h3{font-size:19px}table{font-size:14px}th,td{padding:9px 8px}.widget{padding:15px}.widget table select{font-size:12px}pre{padding:13px} .controls{align-items:flex-start}}
@media print{@page{size:A4;margin:17mm}html{scroll-behavior:auto}body{background:white;color:#222;font-size:11pt;line-height:1.55}.shell{display:block;max-width:none;padding:0}.sidebar,.toolbar,.controls,.no-print{display:none!important}.cover,section{border:0;border-radius:0;padding:0;margin-bottom:24pt}.cover{break-after:page}.cover h1{font-size:31pt}.cover .subtitle{font-size:14pt}h2{font-size:19pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}p{orphans:3;widows:3}table{font-size:9pt}thead{display:table-header-group}tr,figure,.widget{break-inside:avoid}th,td{padding:7px}pre{font-size:9pt;break-inside:avoid}pre code{font-size:9pt}details:not([open])>:not(summary){display:block!important}details{break-inside:auto}summary{break-after:avoid}figure img{max-height:220mm;object-fit:contain}a{color:#263b2b;text-decoration:none}.key,.note,.widget,.equation{background:#f7f8f4!important;print-color-adjust:exact}.foot{font-size:9pt}}
'''

CSS = CSS.replace('@media(max-width:550px){body{font-size:17px}', '@media(max-width:550px){:root{--text:17px}')
CSS = CSS.replace('Georgia,"Times New Roman",serif', '"Times New Roman",serif')

JAVASCRIPT = r'''
(function(){
  const labels=['Không hợp lệ','Input hợp lệ, expected sai','Expected đúng, chưa bắt lỗi','Expected đúng và bắt lỗi'];
  const rewardValues=()=>document.getElementById('reward-profile').value==='equal'?[0,1/3,2/3,1]:[0,.1,.3,1];
  const fmt=x=>Math.abs(x)<1e-10?'0':x.toFixed(3).replace('.',',');
  function updateAdvantages(){
    const rewardMap=rewardValues();const values=Array.from(document.querySelectorAll('.result-category')).map(s=>rewardMap[Number(s.value)]);
    const mean=values.reduce((a,b)=>a+b,0)/4;const std=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/3);const adv=values.map(x=>(x-mean)/(std+.0001));
    document.querySelectorAll('#adv-rows tr').forEach((row,i)=>{row.querySelector('.reward-value').textContent=fmt(values[i]);row.querySelector('.adv-value').textContent=fmt(adv[i]);});
    document.getElementById('adv-summary').textContent='Trung bình = '+fmt(mean)+'; độ lệch chuẩn mẫu = '+fmt(std)+'. '+(std<1e-10?'Các advantage đều bằng 0: nhóm không tạo tín hiệu phân biệt từ reward.':'Advantage dương: tốt hơn trung bình; âm: thấp hơn trung bình theo reward của nhóm.');
    document.getElementById('adv-bars').innerHTML=adv.map((a,i)=>{const w=Math.min(48,Math.abs(a)/1.6*48);return '<div class="bar-row"><span>Mẫu '+String.fromCharCode(65+i)+'</span><div class="bar-track"><span class="bar-fill '+(a<0?'negative':'')+'" style="left:'+(a<0?50-w:50)+'%;width:'+w+'%"></span></div><b>'+fmt(a)+'</b></div>';}).join('');
  }
  function updateCase(){
    const n=Number(document.getElementById('case-n').value), y=Number(document.getElementById('case-y').value);const out=document.getElementById('case-answer');
    if(!document.getElementById('case-complete').checked){out.textContent='Đầu ra chưa hoàn tất/không đúng JSON theo mô phỏng → không hợp lệ, reward 0.';return;}
    if(document.getElementById('case-n').value===''||!Number.isSafeInteger(n)||n<0||n>8){out.textContent='n phải là số nguyên từ 0 đến 8 → input không hợp lệ, reward 0.';return;}
    const gold=n*(n+1)/2, actual=n*(n-1)/2;
    const correct=document.getElementById('case-y').value!==''&&Number.isFinite(y)&&y===gold;const hit=correct&&gold!==actual;const r=hit?1:correct?.3:.1;
    out.textContent='n = '+n+' → reference = '+gold+', code lỗi = '+actual+'. '+(!correct?'Expected đề xuất sai.':hit?'Expected đúng và phát hiện lỗi.':'Expected đúng, code chưa bị phân biệt ở input này.')+' Reward hiện tại = '+fmt(r)+'.';
  }
  document.querySelectorAll('.result-category').forEach(e=>e.addEventListener('change',updateAdvantages));document.getElementById('reward-profile').addEventListener('change',updateAdvantages);
  document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>{const preset={mixed:[0,1,2,3],correct:[2,2,2,2],hit:[3,3,3,3],invalid:[0,0,0,0]}[b.dataset.preset];document.querySelectorAll('.result-category').forEach((e,i)=>e.value=String(preset[i]));updateAdvantages();}));
  ['case-n','case-y','case-complete'].forEach(id=>document.getElementById(id).addEventListener('input',updateCase));
  let textSize=parseFloat(getComputedStyle(document.body).fontSize);document.querySelectorAll('[data-font]').forEach(b=>b.addEventListener('click',()=>{textSize=Math.max(15,Math.min(24,textSize+Number(b.dataset.font)));document.documentElement.style.setProperty('--text',textSize+'px');}));
  document.getElementById('open-answers').addEventListener('click',()=>{const all=Array.from(document.querySelectorAll('details.question,details.deep'));const open=all.some(d=>!d.open);all.forEach(d=>d.open=open);document.getElementById('open-answers').textContent=open?'Thu gọn giải thích':'Mở toàn bộ giải thích';});
  document.getElementById('print-document').addEventListener('click',()=>window.print());
  let printState=[];window.addEventListener('beforeprint',()=>{printState=Array.from(document.querySelectorAll('details')).map(d=>[d,d.open]);printState.forEach(([d])=>d.open=true);});window.addEventListener('afterprint',()=>printState.forEach(([d,v])=>d.open=v));
  if(window.matchMedia('(max-width:950px)').matches)document.getElementById('toc-box').open=false;
  if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){document.querySelectorAll('.sidebar a').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+entry.target.id));}});},{rootMargin:'-8% 0px -70% 0px'});document.querySelectorAll('section').forEach(s=>observer.observe(s));}
  updateAdvantages();updateCase();
})();
'''


def main():
    state = read(PROJECT / "APPS_STATE.json")
    if state["dataset_id"] != "apps-intro-call-v2-eb69b8869857":
        raise ValueError("Review lesson text before changing dataset")
    root = PROJECT / "results/apps-reward-ablation"
    summary = read(root / "metrics.json")
    sampling = read(root / "sampling-metrics.json")
    if summary["trial"]["status"] != "completed" or sampling["sampling_result"]["status"] != "completed":
        raise ValueError("Completed evidence required")
    equal = summary["trial"]["arms"]["equal-v1"]["validation_metrics"]
    if [equal[key] for key in ("generated_tests", "valid_tests", "correct_expected", "bug_detecting_tests", "cases_detected")] != [60, 45, 15, 5, 4]:
        raise ValueError("Update fixed worked metric example before rebuilding")
    train = [json.loads(line) for line in (ASSETS / "data/apps-introductory-v2/train.jsonl").read_text(encoding="utf-8").splitlines()]
    case = next(row for row in train if row["source_id"] == 3019)
    candidates = [json.loads(line) for line in (ASSETS / "evidence/apps-adversary-20261028-160208-467069/adversarial-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    adversary = next(row for row in candidates if row["candidate_id"] == "r1-3019-08")
    if [adversary["gate"][key] for key in ("accepted", "domain_inputs", "wrong_inputs")] != [True, 363, 90]:
        raise ValueError("Adversary example changed")
    app_example = '<p>Case: <code>' + html.escape(case["case_id"]) + '</code>. <code>strng</code> dài 0–4; <code>letter</code> dài đúng 1; ký tự được phép là a, b, A.</p><div class="two"><div><h3>Reference nguồn</h3>' + code(case["reference"]) + '</div><div><h3>Mutant AST m1</h3>' + code(case["buggy_code"]) + '</div></div><p>Mutant ban đầu đổi điều kiện so sánh. Code đối kháng ở mục 12 là một bản khác do model local sinh.</p>'
    adaptive = csv_rows(PROJECT / "results/apps-adversarial/metrics.csv")
    adaptive_table = table(["Mốc", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code /15"], [[r["label"],r["valid"],r["correct_expected"],r["bug_detecting_tests"],r["cases_detected"]] for r in adaptive], highlight=True)
    sft = csv_rows(PROJECT / "results/apps-sft-budget/metrics.csv")
    selected = {r["variant"]: r for r in sft}
    sft_table = table(["Mốc cùng lượt mới", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code /15"], [[label,*[selected[key][f] for f in ("valid_tests","correct_expected","bug_tests","cases_detected")]] for key,label in (("SFT64_same_trajectory","SFT64 cùng quỹ đạo"),("SFT256","SFT256"))], highlight=True)
    reward_table = table(["Mốc, seed sinh 42", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code /15"], [[label,*[m[f] for f in ("valid_tests","correct_expected","bug_detecting_tests","cases_detected")]] for label,m in (("SFT64 lịch sử",summary["trial"]["parent_validation_metrics"]),("GRPO reward hiện tại",summary["trial"]["arms"]["legacy-v1"]["validation_metrics"]),("GRPO reward chia đều",equal))], highlight=True)
    indexed = {(r["variant"],r["generation_seed"]):r["metrics"] for r in sampling["sampling_result"]["runs"]}
    seed_table = table(["Code phát hiện /15", "Seed sinh 42", "Seed sinh 43", "Seed sinh 44"], [[label,*[indexed[(key,seed)]["cases_detected"] for seed in (42,43,44)]] for key,label in (("SFT64_parent","SFT64 trước RL"),("legacy-v1","GRPO reward hiện tại"),("equal-v1","GRPO reward chia đều"))], highlight=True)
    env = read(PROJECT / "results/apps-adversarial/environment.json")
    env_rows = [["CPU",env["cpu"]],["GPU / driver",env["gpu_memory_driver"]],["RAM",f'{env["physical_ram_bytes"]/2**30:.2f} GiB, khoảng 16 GB'],["Python",env["python"].split(' (')[0]],["Model",env["model_repository"]]] + [[name,version] for name,version in env["libraries"].items()]
    values = [0,.1,.3,1]
    mean, std = statistics.mean(values), statistics.stdev(values)
    advantages = [(value-mean)/(std+.0001) for value in values]
    category_labels = ["Không hợp lệ","Input hợp lệ, expected sai","Expected đúng, chưa bắt lỗi","Expected đúng và bắt lỗi"]
    adv_rows = []
    for i in range(4):
        options = ''.join('<option value="'+str(j)+'"'+(' selected' if i==j else '')+'>'+label+'</option>' for j,label in enumerate(category_labels))
        adv_rows.append('<tr><td>'+chr(65+i)+'</td><td><select class="result-category" aria-label="Loại kết quả mẫu '+chr(65+i)+'">'+options+'</select></td><td class="reward-value">'+number(values[i])+'</td><td class="adv-value">'+number(advantages[i])+'</td></tr>')
    outcomes = '<figure><svg viewBox="0 0 960 180" role="img" aria-label="60 test gồm 15 không hợp lệ, 30 expected sai, 10 đúng chưa bắt lỗi và 5 bắt lỗi"><g font-family="Segoe UI,Arial,sans-serif" font-size="17">'
    left=40
    for count,label,color in ((15,"Không hợp lệ","#d8ddd2"),(30,"Expected sai","#e4c9ad"),(10,"Đúng, chưa bắt lỗi","#c2d8dd"),(5,"Bắt lỗi","#7fa98b")):
        width=count/60*880
        outcomes += f'<rect x="{left}" y="25" width="{width}" height="48" fill="{color}"/><text x="{left+width/2}" y="56" text-anchor="middle" fill="#243a2b">{count}</text>'
        left += width
    for i,(label,color) in enumerate((("15 không hợp lệ","#d8ddd2"),("30 expected sai","#e4c9ad"),("10 đúng, chưa bắt lỗi","#c2d8dd"),("5 bắt lỗi","#7fa98b"))):
        x=40+i*225
        outcomes += f'<rect x="{x}" y="100" width="16" height="16" fill="{color}"/><text x="{x+24}" y="114" font-size="15" fill="#334538">{label}</text>'
    outcomes += '<text x="480" y="154" text-anchor="middle" fill="#5c6b60" font-size="16">Tất cả 60 mẫu đều ở mẫu số khi tính tỷ lệ theo test.</text></g></svg><figcaption>Bốn nhóm kết quả loại trừ nhau, từ lượt reward chia đều seed 42.</figcaption></figure>'
    glossary = [("Adapter","Phần trọng số điều chỉnh, cần dùng cùng model nền."),("Adversary","Vai trò cung cấp code để thử thách policy sinh test."),("Advantage","Mức tốt/kém tương đối của câu trả lời trong nhóm."),("Ablation","Thử thay hoặc bỏ một thành phần để đo tác động riêng."),("AST","Biểu diễn cấu trúc cú pháp code thành cây."),("Attack Rate","Tỷ lệ test expected đúng và làm lộ lỗi trên tất cả test sinh."),("Baseline","Mốc so sánh để đo phần thay đổi sau can thiệp."),("BF16","Kiểu số floating point 16-bit, khác lượng tử hóa 4-bit."),("Checkpoint","Bản tham số lưu ở một mốc học."),("Completion","Câu trả lời được model sinh sau prompt."),("Curriculum","Tập/diễn tiến dữ liệu dùng để học."),("Epoch","Một lượt tương đương qua tập học, khác optimizer step."),("Expected","Giá trị chương trình đúng phải trả với input đó."),("Fine-tuning","Học thêm model cho nhiệm vụ cụ thể."),("Frozen","Trọng số được giữ cố định trong lượt học."),("Gradient","Thông tin đạo hàm dùng điều chỉnh tham số để tối ưu loss."),("GRPO","Học policy bằng reward tương đối trong nhóm câu trả lời."),("Inference","Sinh kết quả với trọng số hiện tại, không cập nhật tham số."),("IO Accuracy","Tỷ lệ cặp input/expected đúng trên tất cả mẫu sinh."),("JSON / JSONL","Một cấu trúc dữ liệu / nhiều bản ghi JSON, mỗi dòng một bản."),("Learning rate","Hệ số quy mô bước điều chỉnh tham số."),("LoRA","Học phần điều chỉnh hạng thấp khi giữ trọng số nền."),("Loss","Đại lượng tối ưu; ý nghĩa phụ thuộc SFT hay GRPO."),("Mutant","Code được thay đổi có kiểm soát để tạo hành vi sai."),("Observed","Kết quả code đang kiểm tra thật sự trả về."),("Optimizer","Cơ chế dùng gradient và trạng thái để cập nhật tham số."),("Oracle / reference","Chuẩn dùng kiểm tra đáp án trong phạm vi đã xác nhận."),("Policy","Model đang được tối ưu để sinh hành động/câu trả lời."),("Prompt","Nội dung đưa vào model: đề, code và yêu cầu định dạng."),("Reward","Điểm môi trường cho câu trả lời đã sinh."),("Rollout","Lượt sinh câu trả lời và ghi kết quả phục vụ RL."),("Run","Một lượt chạy thực nghiệm có thư mục/cấu hình riêng."),("Seed","Giá trị khởi tạo ngẫu nhiên; cần phân biệt seed train/sinh."),("SFT","Tinh chỉnh có giám sát theo target đúng."),("Step","Một bước đã định nghĩa trong log; thường là cập nhật optimizer."),("Tag","Tên một đại lượng trong TensorBoard."),("Tensor","Mảng số nhiều chiều, không phải một tham số duy nhất."),("Token","Mảnh văn bản model xử lý/sinh."),("Truncated","Đầu ra bị cắt trước khi kết thúc hợp lệ."),("Validation","Tập phát triển dùng so thiết kế, khác Train/Test cuối.")]
    replacements = {"__PIPELINE__":PIPELINE,"__LORA__":LORA,"__APPS_EXAMPLE__":app_example,"__ADVERSARY_CODE__":code(adversary["buggy_code"]),"__ADV_ROWS__":''.join(adv_rows),"__ADV_INITIAL__":'Trung bình = '+number(mean)+'; độ lệch chuẩn mẫu = '+number(std)+'. Advantage theo thứ tự: '+', '.join(number(x) for x in advantages)+'.',"__ADV_VALUES__":'; '.join(number(x) for x in advantages),"__ADV_BARS__":'<p class="count">Các thanh advantage xuất hiện khi mở bằng trình duyệt có JavaScript.</p>',"__OUTCOMES__":outcomes,"__ADAPTIVE_TABLE__":adaptive_table,"__SFT_TABLE__":sft_table,"__REWARD_TABLE__":reward_table,"__SEED_TABLE__":seed_table,"__ENV_TABLE__":table(["Thành phần","Snapshot thực nghiệm"],env_rows),"__TB_SCREENSHOT__":image(SOURCE/'tensorboard.png','TensorBoard nền sáng với bộ lọc run, smoothing và biểu đồ verified attack/IO trên Train'),"__SFT_FIGURE__":image(PROJECT/'results/apps-sft-budget/validation-comparison.png','So sánh SFT64 và SFT256 cùng lượt học'),"__SEED_FIGURE__":image(root/'sampling-variability.png','Kết quả ba seed sinh test trên cùng checkpoint'),"__GLOSSARY__":table(["Thuật ngữ","Nghĩa trong đồ án"],glossary)}
    expected_review = read(PROJECT / "results/apps-input-first-review/metrics.json")
    expected_labels = {"base":"Qwen gốc", "sft":"SFT64 input-first", "control":"GRPO cố định · 2 vòng", "adaptive":"GRPO đối kháng · 2 vòng"}
    replacements["__EXPECTED_TABLE__"] = table(["Checkpoint", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code /15"], [[expected_labels[name]] + [f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}' for f in ("valid_tests","correct_expected","bug_detecting_tests","cases_detected")] for name,values in expected_review["summary"].items()])
    replacements["__EXPECTED_ERRORS__"] = image(PROJECT / "results/apps-expected-diagnosis/expected-errors.png", "Phân loại đầu ra của năm checkpoint trên ba seed sinh Validation")
    budget = read(PROJECT / "results/apps-shared-sft-budget/metrics.json")
    replacements["__SHARED_BUDGET_TABLE__"] = table(["SFT cùng lượt", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code /15"], [[steps + " bước"] + [f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}' for f in ("valid_tests","correct_expected","bug_detecting_tests","cases_detected")] for steps,values in budget["summary"].items()])
    followup = read(PROJECT / "results/apps-sft256-grpo/metrics.json")
    replacements["__SFT256_RL_TABLE__"] = table(["Checkpoint", "Hợp lệ /60", "Expected đúng /60", "Test bắt lỗi /60", "Code /15"], [["SFT256" if name == "sft256" else "SFT256 + GRPO96"] + [f'{values[f]["mean"]:.2f} ± {values[f]["sample_sd"]:.2f}' for f in ("valid_tests","correct_expected","bug_detecting_tests","cases_detected")] for name,values in followup["summary"].items()])
    fragments = sorted(SOURCE.glob('0[1-5]-*.html'))
    if len(fragments)!=5:
        raise ValueError("Five teaching parts required")
    body='\n'.join(path.read_text(encoding='utf-8') for path in fragments)
    for key,value in replacements.items():
        if key not in body:
            raise ValueError('Unused/missing placeholder: '+key)
        body=body.replace(key,value)
    if re.search(r'__[A-Z_]+__',body):
        raise ValueError('Unresolved lesson placeholder')
    headings=re.findall(r'<section id="([^"]+)">.*?<h2>(.*?)</h2>',body,re.S)
    if len(headings)!=24 or len({key for key,_ in headings})!=24:
        raise ValueError('Expected 24 unique teaching sections')
    toc=''.join('<a href="#'+key+'">'+title+'</a>' for key,title in headings)
    header='''<header class="cover" id="start"><p class="kicker">Bài giảng cá nhân · Đồ án TTNT số 19</p><h1>Hiểu đồ án sinh test bằng AI</h1><p class="subtitle">Từ một input/expected đến SFT, GRPO, đối kháng và TensorBoard.<br>Adversarial Reinforcement Learning for Test Case Generation</p><p>Viết để đọc độc lập, bắt đầu từ ví dụ nhỏ rồi nối với đúng dữ liệu, phần mềm và kết quả của project. Không cần mở slide hoặc tự đối chiếu log trong lúc đọc.</p><div class="toolbar"><button type="button" data-font="-1" aria-label="Giảm cỡ chữ">A−</button><button type="button" data-font="1" aria-label="Tăng cỡ chữ">A+</button><button type="button" id="open-answers">Mở toàn bộ giải thích</button><button type="button" id="print-document">In / lưu PDF</button></div><div class="read-routes"><p><strong>Cập nhật kết quả:</strong> mục24 giải thích expected sai, nhãn input-first và lượt đối kháng mới; các ví dụ thực nghiệm cũ giữ mốc riêng.</p><p><strong>Lượt 1 — nắm bài toán:</strong> mục 1–6, rồi 12–14. Tự chấm được một test và nói hệ thống đang học gì.</p><p><strong>Lượt 2 — hiểu cơ chế:</strong> mục 7–11. Thử mô phỏng GRPO; phần toán mở rộng có thể đọc sau.</p><p><strong>Lượt 3 — dùng được công cụ:</strong> mục 15–18, rồi trả lời câu hỏi ở 20–21. Ưu tiên hiểu bằng lời của mình.</p></div><p class="meta">Cập nhật 30/10/2026 · 24 mục · 2 mô phỏng tính toán · Hình/ảnh nhúng sẵn · Nền sáng · Đọc offline</p><noscript><p class="no-js">Bạn vẫn đọc được toàn bộ phần giảng và ví dụ số. Hai mô phỏng cần JavaScript để thay đổi tham số.</p></noscript></header>'''
    document='<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="Bài giảng dễ hiểu về project sinh test bằng Qwen local, LoRA, SFT, GRPO, đối kháng và TensorBoard."><title>Bài giảng đồ án TTNT — Sinh test bằng học tăng cường đối kháng</title><style>'+CSS+'</style></head><body><div class="shell"><aside class="sidebar"><small>Nguyễn Quốc An · 24DTHC1<br>Tài liệu tự học, chưa phải báo cáo nộp.</small><nav aria-label="Mục lục"><details id="toc-box" open><summary>Mục lục · 24 mục</summary><a href="#start">Cách đọc tài liệu</a>'+toc+'</details></nav></aside><main>'+header+body+'<p class="foot">File tự chứa. Dùng Ctrl+F để tìm thuật ngữ; nguồn trực tuyến chỉ được mở khi bạn bấm liên kết.</p></main></div><script>'+JAVASCRIPT+'</script></body></html>'
    OUTPUT.write_text(document,encoding='utf-8',newline='\n')
    manifest={"date":"2026-10-30","output":str(OUTPUT),"bytes":OUTPUT.stat().st_size,"sections":len(headings),"self_contained_images":len(re.findall(r'<img\b',document)),"inline_svg_figures":len(re.findall(r'<svg\b',document)),"model_training_started":False,"model_test_evaluation":False,"source_files":{path.relative_to(PROJECT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in fragments+[Path(__file__),SOURCE/'tensorboard.png']},"reward_trial_result_sha256":summary['trial_result_sha256'],"sampling_runs":len(sampling['sampling_result']['runs']),"metric_example":[60,45,15,5,4]}
    (SOURCE/'build-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:manifest[k] for k in ('output','bytes','sections','sampling_runs','model_training_started')},ensure_ascii=False))


if __name__=='__main__':
    main()
