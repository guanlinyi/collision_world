"""Build an offline purpose-to-code story; no models, network or source imports.

Run from any directory. Existing output is protected by exclusive creation.
Content lives in input_json/story.json. Code excerpts come from Python AST nodes;
data cards read only explicitly selected fields from the repository's evidence.
"""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
DEFAULT = HERE / 'output_html/从目的到代码_故事版.html'


def read(relative):
    return json.loads((PROJECT / relative).read_text(encoding='utf-8-sig'))


def esc(value):
    return html.escape(str(value), quote=True)


def source(spec):
    path = PROJECT / spec['path']
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    tree = ast.parse(text)
    nodes = [tree]
    for name in spec['symbol'].split('.'):
        nodes = [child for node in nodes for child in getattr(node, 'body', [])
                 if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and child.name == name]
    if not nodes:
        raise ValueError('AST symbol missing: ' + str(spec))
    node = nodes[-1]  # Python's last definition wins when a historical file redefines it.
    first, last = node.lineno, node.end_lineno
    if 'slice' in spec:
        lo, hi = spec['slice']
        first, last = first + lo, min(first + hi, last)
        if first > last:
            raise ValueError('Invalid AST-relative excerpt')
    code = '\n'.join(text.splitlines()[first - 1:last])
    return dict(path=spec['path'], symbol=spec['symbol'], first=first, last=last,
                file_sha256=hashlib.sha256(raw).hexdigest(),
                excerpt_sha256=hashlib.sha256(code.encode()).hexdigest(), code=code)


def code_card(record):
    return ('<details class="source"><summary>看这一步的代码：' + esc(record['symbol']) +
            '</summary><p class="source-meta">' + esc(record['path']) +
            ' · 实际第 ' + str(record['first']) + '–' + str(record['last']) +
            ' 行。</p><pre><code>' + esc(record['code']) +
            '</code></pre><details class="verification"><summary>核对代码摘要</summary><p class="hash">文件 SHA-256：' +
            record['file_sha256'] + '<br>片段 SHA-256：' + record['excerpt_sha256'] + '</p></details></details>')


def data_card(title, value, paths):
    return '<div class="data"><h3>' + esc(title) + '</h3><pre>' + esc(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre><p class="source-meta">数值来源：' + '；'.join(esc(p) for p in paths) + '</p></div>'


def evidence_cards():
    safe = read('estimate_geometry/refine_contours/input_json/safe_inputs.json')
    obs = read('estimate_geometry/refine_contours/input_json/observation.json')
    prompt = read('estimate_geometry/observe_structure/input_json/frame28_inputs.json')
    accept = read('docs/validation/output_json/latest014_frame28.json')
    sam = read('docs/validation/output_json/sam_cache_selection.json')
    poses = read('datasets/carla_replay/case07/input_json/pose_experiments.json')['pose_experiments']
    latest = next(m for m in poses if m['id'] == 'case07_far_exterior_bt8')
    pair07 = read('datasets/carla_replay/case07/input_json/pair_geometry.json')['records']
    pair11 = read('datasets/carla_replay/case11/input_json/pair_geometry.json')['records']
    cards = {}
    cards['prompts'] = data_card('本帧真正交给前端的输入（像素坐标）',
        {k: prompt[k] for k in ['frame', 'car', 'crop_xyxy', 'positive_points_crop_xy', 'negative_points_crop_xy', 'body_box_crop_xyxy', 'NCC_score']},
        ['estimate_geometry/observe_structure/input_json/frame28_inputs.json'])
    cards['sam'] = data_card('缓存候选选择的验收记录', {k: sam[k] for k in ['selected', 'selected_area_pixels', 'selected_mask_pixel_identical', 'SAM_forward_rerun']}, ['docs/validation/output_json/sam_cache_selection.json'])
    example = next(r for r in obs['records'] if r['status'] == 'photometric_candidate')
    cards['edges'] = data_card('463 个轮廓点里，248 个成为 RGB 边候选',
        dict(total_candidates=obs['total_candidates'], photometric_candidates=obs['photometric_candidates'], unknown=obs['unknown'],
             first_accepted_record={k: example[k] for k in ['id', 'SAM_seed_full_image_xy', 'outward_normal_xy', 'main_peak_full_image_xy', 'main_peak_offset_px', 'secondary_main_ratio', 'status']}),
        ['estimate_geometry/refine_contours/input_json/observation.json'])
    cards['prior'] = data_card('起点与边界来自给定输入',
        dict(baseline_pose=safe['baseline_pose'], old_predicted_prior=safe['old_predicted_prior'],
             intrinsics=safe['calibration']['intrinsics'], query_truth_read=safe['query_truth_read']),
        ['estimate_geometry/refine_contours/input_json/safe_inputs.json'])
    cards['result'] = data_card('这一次 CPU 重算留下的结果',
        {k: accept[k] for k in ['elapsed_seconds', 'status', 'accepted_outer_rounds', 'candidate_pose', 'candidate_cost', 'center_difference_m', 'rotation_difference_deg', 'cost_difference', 'reference_truth_read']},
        ['docs/validation/output_json/latest014_frame28.json'])
    gap_rows = []
    for case, records, method, frames in [('07', pair07, 'case07_far_exterior_bt8', [28, 40, 41]), ('11', pair11, 'contiguous_exact_contour_bt8', [22])]:
        for f in frames:
            r = records[f]
            gap_rows.append(dict(case=case, frame=f, method=method, predicted_distance_m=r['methods'][method]['distance_m'], reference_distance_m=r['reference']['distance_m'], predicted_contact=r['methods'][method]['contact'], reference_contact=r['reference']['contact']))
    cards['gaps'] = data_card('网页已有冻结距离：同一时刻的两车一起算', gap_rows,
        ['datasets/carla_replay/case07/input_json/pair_geometry.json', 'datasets/carla_replay/case11/input_json/pair_geometry.json'])
    rows = []
    for f in [18, 19, 20]:
        r = next(r for r in latest['records'] if r['frame'] == f and r['car'] == 0)
        rows.append(dict(frame=f, x_m=r['pose'][0], y_m=r['pose'][1], center_error_m=r['metrics']['center_error_m'], velocity_vector_mps=r['metrics']['velocity_vector_lab_mps'], velocity_error_mps=r['metrics']['velocity_error_mps']))
    cards['velocity'] = data_card('车0从第18帧进入第19帧：位置差分变成了速度', rows,
        ['datasets/carla_replay/case07/input_json/pose_experiments.json · case07_far_exterior_bt8'])
    counter = read('docs/program_flow/input_json/f19_counterexample.json')
    cards['counterexample'] = data_card('已保存第19帧接受路径的评价摘录（中心误差单位：厘米）',
        dict(accepted_rounds=counter['accepted_rounds'],
             fixed_anchor_count=counter['saved_cost_decomposition']['fixed_anchor_count'],
             fixed_weights=counter['saved_cost_decomposition']['fixed_weights'],
             center_error_requires_reference_truth=counter['center_error_requires_reference_truth'],
             sources=counter['sources']), ['docs/program_flow/input_json/f19_counterexample.json'])
    return cards


CSS = r'''
:root{--bg:#f6f4ee;--paper:#fffefa;--ink:#24322e;--muted:#65716a;--line:#dcded5;--given:#355fa3;--obs:#087d76;--fit:#9a5a19;--eval:#7853a2;--warn:#ab493b}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:92px}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.9 'Segoe UI','Microsoft YaHei',sans-serif}a{color:var(--obs);text-decoration-thickness:1px;text-underline-offset:3px}aside{position:fixed;inset:0 auto 0 0;width:294px;background:#eef0e9;border-right:1px solid var(--line);padding:28px 19px;overflow:auto}aside h2{font-size:19px;line-height:1.5;margin:0 0 9px}.intro{font-size:13px;color:var(--muted);line-height:1.7}nav a{display:block;font-size:14px;line-height:1.6;padding:7px 9px;border-left:3px solid transparent;text-decoration:none;color:#46564c}nav a:hover,nav a.active{background:#fffefa;border-color:var(--obs);color:#126961}nav details{margin:5px 0}nav summary{cursor:pointer;font-weight:600;font-size:14px}nav .child{padding-left:17px}main{max-width:1160px;margin-left:294px;padding:35px 60px 100px}.topbar{position:sticky;top:0;z-index:3;background:rgba(246,244,238,.97);border-bottom:1px solid var(--line);padding:10px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px}.topbar a,.up a,.prevnext a,button{border:1px solid var(--line);background:var(--paper);padding:5px 11px;border-radius:7px;color:#3d5547;text-decoration:none;cursor:pointer;font:inherit}.topbar span{color:var(--muted);margin-left:auto}section{padding:42px 0 34px;border-bottom:1px solid var(--line)}section:first-of-type{padding-top:24px}.crumb{font-size:13px;color:var(--muted);letter-spacing:.06em}h1{font-size:40px;line-height:1.45;letter-spacing:-.02em;margin:16px 0 24px}h2{font-size:29px;line-height:1.5;margin:10px 0 23px}h3{font-size:19px;line-height:1.6;margin:24px 0 10px}p{margin:14px 0}.lead{font-size:21px;line-height:1.85}.kicker{font-size:13px;letter-spacing:.1em;font-weight:700;color:var(--obs)}.legend{display:flex;flex-wrap:wrap;gap:9px;margin:22px 0}.badge{padding:3px 11px;border-radius:30px;font-size:13px;background:#e8edeb;border:1px solid currentColor}.given{color:var(--given)}.observed{color:var(--obs)}.fitted{color:var(--fit)}.evaluated{color:var(--eval)}.warn{border-left:4px solid var(--warn);padding:13px 18px;background:#fff0e9}.note{border-left:4px solid var(--obs);padding:12px 18px;background:#edf5ef}.flow{background:#eaf0ea;border:1px solid #d2dfd5;border-radius:14px;padding:22px;margin:26px 0}.flow h3{margin-top:0}.flow-node{display:block;background:var(--paper);padding:16px 18px;border:1px solid #cbd8cf;border-left:5px solid var(--obs);border-radius:8px;text-decoration:none;color:var(--ink)}.flow-node strong{display:block;font-size:18px}.flow-node span{display:block;color:var(--muted);font-size:14px;margin-top:4px}.arrow{text-align:center;font-size:21px;line-height:1.5;color:#83978b}.branch{display:grid;grid-template-columns:1fr 1fr;gap:15px}.branch>div{padding:15px 19px;background:var(--paper);border:1px solid var(--line);border-radius:10px}.branch h3{margin-top:0}.data{border-left:4px solid var(--given);background:#edf2f8;padding:1px 19px 12px;margin:25px 0}.data h3{color:#31537b}.data pre{max-height:470px;overflow:auto;background:rgba(255,255,255,.6);font-size:13px;color:#263f55;padding:12px}.source{margin:26px 0;border:1px solid var(--line);border-radius:9px;background:var(--paper);overflow:hidden}.source summary{cursor:pointer;padding:13px 17px;font-weight:600;color:#5e4b2c;font-size:15px}.source p{padding:0 18px}.source pre{background:#202e29;color:#e2ece2;margin:0;padding:20px;max-height:650px;overflow:auto;font-size:13px;line-height:1.75}.source-meta{font-size:12px;line-height:1.7;color:var(--muted);overflow-wrap:anywhere}.hash{font-size:11px;font-family:Consolas,monospace;word-break:break-all;color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:Consolas,'Microsoft YaHei',monospace}p code,td code{font-size:.89em;background:#e7eae2;padding:2px 4px;border-radius:4px;overflow-wrap:anywhere}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.8;margin:22px 0}th,td{border:1px solid var(--line);padding:12px 13px;text-align:left;vertical-align:top}th{background:#e9ede5}td{background:#fffefa}.up{font-size:13px;margin-top:26px;display:flex;gap:9px}.prevnext{display:flex;justify-content:space-between;gap:13px;font-size:13px;margin-top:30px}.diagram{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;overflow:auto}.diagram svg{display:block;max-width:100%;min-width:430px;height:auto}.caption{font-size:13px;color:var(--muted)}.run{background:#26372e;color:#e9f0e9;padding:18px;font-size:13px;border-radius:8px}.milestones{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0}.milestones div{background:#fffefa;border-top:3px solid var(--obs);padding:14px;font-size:14px}.milestones b{display:block;font-size:25px}.hidden{display:none!important}#search{width:100%;padding:10px;border:1px solid #c6d1c4;border-radius:7px;background:#fffefa;margin:12px 0;font:inherit;font-size:14px}@media(max-width:900px){aside{position:relative;width:auto;max-height:370px;border-bottom:1px solid var(--line)}main{margin-left:0;padding:20px 24px 70px}h1{font-size:31px}h2{font-size:25px}.branch,.milestones{grid-template-columns:1fr}.topbar span{margin-left:0}section{padding-top:30px}}@media print{aside,.topbar,.up,.prevnext{display:none}main{margin:0;max-width:none;padding:0}details{break-inside:avoid}body{background:white;font-size:12px}h1{font-size:26px}h2{font-size:21px}section{padding:15px 0}}
'''

JS = r'''
const sections=[...document.querySelectorAll('main section')];
const links=[...document.querySelectorAll('nav a')];
const up=document.getElementById('parent-link');
const current=document.getElementById('where');
function mark(id){const s=document.getElementById(id);if(!s)return;links.forEach(a=>a.classList.toggle('active',a.hash==='#'+id));up.href='#'+(s.dataset.parent||'home');current.textContent=s.dataset.short;const a=links.find(a=>a.hash==='#'+id);for(let p=a?.parentElement;p;p=p.parentElement){if(p.tagName==='DETAILS')p.open=true}}
function showChapter(){const id=location.hash.slice(1)||'home';const target=document.getElementById(id);if(!target||target.tagName!=='SECTION')return;document.body.classList.add('browse');sections.forEach(s=>s.classList.toggle('current',s===target));mark(id);window.scrollTo({top:0,behavior:'instant'})}
window.addEventListener('hashchange',showChapter);
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();links.forEach(a=>a.classList.toggle('hidden',q&&!a.textContent.toLowerCase().includes(q)));document.querySelectorAll('nav details').forEach(d=>{if(q)d.open=true})});
document.getElementById('code-toggle').addEventListener('click',e=>{const ds=[...document.querySelectorAll('details.source')];const open=ds.some(d=>!d.open);ds.forEach(d=>d.open=open);e.target.textContent=open?'收起全部代码':'展开全部代码'});
showChapter();
'''


def build():
    content = json.loads((HERE / 'input_json/story.json').read_text(encoding='utf-8'))
    excerpts = {key: source(spec) for key, spec in content['sources'].items()}
    cards = evidence_cards()
    sections = content['sections']
    by_id = {s['id']: s for s in sections}
    if len(by_id) != len(sections):
        raise ValueError('Duplicate chapter id')
    def flow(ids):
        return '<div class="flow"><h3>接下来，我们把这件事拆开看</h3>' + '<div class="arrow" aria-hidden="true">↓</div>'.join(
            '<a class="flow-node" href="#' + sid + '"><strong>' + esc(by_id[sid]['title']) + '</strong><span>' + esc(by_id[sid]['teaser']) + '　点开这一步 →</span></a>' for sid in ids) + '</div>'
    def replace(match):
        kind, key = match.group(1), match.group(2)
        return code_card(excerpts[key]) if kind == 'code' else cards[key]
    nav = '<a href="#home">总图｜我们到底想判断什么</a>'
    for s in sections[1:]:
        if s['parent'] == 'home':
            children = [c for c in sections if c.get('parent') == s['id']]
            nav += '<details open><summary>' + esc(s['short']) + '</summary><a href="#' + s['id'] + '">本层总图</a>'
            nav += ''.join('<a class="child" href="#' + c['id'] + '">' + esc(c['short']) + '</a>' for c in children) + '</details>'
    body = ''
    for i, s in enumerate(sections):
        rendered = re.sub(r'\[\[(code|data):([\w]+)\]\]', replace, s['body'])
        if re.search(r'\[\[(code|data):', rendered):
            raise ValueError('Unresolved content placeholder')
        if s.get('flow'):
            rendered += flow(s['flow'])
        parent = s.get('parent', 'home')
        rendered += '<div class="up"><a href="#' + parent + '">↑ 回到' + ('总图' if parent == 'home' else '上一级：' + esc(by_id[parent]['short'])) + '</a><a href="#home">↖ 回到总图</a></div>'
        previous = sections[max(0, i - 1)]
        following = sections[min(len(sections) - 1, i + 1)]
        rendered += '<div class="prevnext"><a href="#' + previous['id'] + '">← ' + esc(previous['short']) + '</a><a href="#' + following['id'] + '">' + esc(following['short']) + ' →</a></div>'
        body += '<section id="' + s['id'] + '" data-parent="' + parent + '" data-short="' + esc(s['short']) + '"><div class="crumb">' + esc(s['crumb']) + '</div><' + ('h1' if i == 0 else 'h2') + '>' + esc(s['title']) + '</' + ('h1' if i == 0 else 'h2') + '>' + rendered + '</section>'
    manifest = [{k: v for k, v in r.items() if k != 'code'} for r in excerpts.values()]
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="沿着一帧真实输入，理解从RGB轮廓到三维位姿、接触与速度核查的代码。"><title>从目的到代码｜两辆车到底有没有接触</title><style>' + CSS + 'body.browse section{display:none}body.browse section.current{display:block}@media print{body.browse section{display:block}}</style></head><body><aside><div class="kicker">COLLISION WORLD · 代码故事</div><h2>两辆车到底有没有接触</h2><p class="intro">先看想解决什么，再看一次输入怎样向下走。蓝＝给定，绿＝图像观测，棕＝求解，紫＝评价。整理于 2026-11-15。</p><input id="search" type="search" aria-label="筛选左侧目录" placeholder="筛选目录，如：速度、轮廓"><nav aria-label="故事目录">' + nav + '</nav><p class="intro">单文件离线阅读，无外部脚本或字体。所有代码可展开；网页不会运行模型。</p></aside><main><div class="topbar"><a href="#home">总图</a><a id="parent-link" href="#home">↑ 上一级</a><button id="code-toggle" type="button">展开全部代码</button><span id="where">故事起点</span></div>' + body + '</main><script type="application/json" id="source-manifest">' + json.dumps(manifest, ensure_ascii=False).replace('<', '\\u003c') + '</script><script>' + JS + '</script></body></html>'
    ids = set(re.findall(r'\bid="([^"]+)"', document))
    missing = set(re.findall(r'href="#([^"]+)"', document)) - ids
    if missing:
        raise ValueError('Broken chapter links: ' + str(missing))
    if len(document.encode('utf-8')) >= 500_000:
        raise ValueError('Story exceeds 500KB budget')
    return document, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT, help='New HTML path within docs/program_flow; never overwritten')
    parser.add_argument('--check-only', action='store_true', help='Validate current source/data and render in memory, without writing')
    args = parser.parse_args()
    document, manifest = build()
    if not args.check_only:
        output = args.output.resolve()
        if not output.is_relative_to(HERE):
            raise ValueError('Output must stay within docs/program_flow')
        if output.suffix.lower() != '.html':
            raise ValueError('Output extension must be .html')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(document)
        print('Created', output)
    print(json.dumps(dict(bytes=len(document.encode('utf-8')), excerpts=len(manifest),
                         source_and_internal_links_verified=True, models_executed=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
