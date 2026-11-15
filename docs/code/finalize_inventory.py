"""Freeze public file hashes and perform final static integrity checks; no publish."""
import argparse,ast,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def included(path):
    r=path.relative_to(ROOT)
    return ('experiments' not in r.parts and '__pycache__' not in r.parts and
            not any(p.startswith('browser_profile') for p in r.parts) and
            'output_txt' not in r.parts and path.suffix not in ['.pyc','.log'] and
            not path.name.endswith('_20261115.md'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Output outside project')
    files=[f for f in ROOT.rglob('*') if f.is_file() and included(f) and f.name!='SOURCE_MANIFEST.json' and f!=output]
    # Child SOURCE_MANIFEST files remain part of the source-group map below.
    files.extend(f for f in ROOT.rglob('SOURCE_MANIFEST.json') if f.parent!=ROOT and included(f))
    files=sorted(set(files));errors=[];py=0;links=0;drive=[];credentials=[]
    for file in files:
        if file.suffix=='.py':
            ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file));py+=1
        if file.suffix in ['.py','.js','.html','.css']:
            s=file.read_text(encoding='utf-8-sig')
            if re.search(r'(?<![A-Za-z])[EFG]:[\\/]',s):drive.append(file.relative_to(ROOT).as_posix())
        if file.suffix in ['.py','.js','.json','.md','.html','.css']:
            s=file.read_text(encoding='utf-8-sig')
            if re.search(r'BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}',s):credentials.append(file.relative_to(ROOT).as_posix())
        if file.suffix=='.md':
            for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',file.read_text(encoding='utf-8-sig')):
                target=match.group(1).strip('<>').split('#')[0]
                if not target or re.match(r'^[a-z]+://',target):continue
                links+=1
                if not (file.parent/target).exists():errors.append(dict(file=str(file.relative_to(ROOT)),target=target))
    records=[dict(path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]
    manifest=dict(schema='collision_public_source_manifest_v1',date='2026-11-15',
        status='Local review candidate, not published; original rights retained',
        hash_authority='files contains final current public-file hashes. Older supplemental transfer receipts retain their initial target hashes; primary_sources has final copied-code/data hashes.',
        source_groups=['docs/provenance/primary_sources.json','docs/provenance/additional_function_sources.json',
            'docs/provenance/personal_checkpoint_source_manifest.json','learn_relations/SOURCE_MANIFEST.json',
            'learn_relations/SOURCE_MANIFEST_TARGETS.json','datasets/record_carla/SOURCE_MANIFEST_STATIC_EXPORT.json',
            'datasets/carla_surface_sample/prepared_sample/input_json/SOURCE_SELECTION.json',
            'estimate_geometry/refine_contours/SOURCE_MANIFEST.json','estimate_geometry/refine_contours/SOURCE_TRANSFORMATION.json',
            'estimate_geometry/observe_structure/SOURCE_MANIFEST.json'],
        exclusions='All experiments, browser profiles/logs/cache, Python bytecode, local progress journals; this root manifest and final audit itself excluded from recursive file hashes.',
        files=records)
    for group in manifest['source_groups']:
        assert (ROOT/group).is_file(),group
    with (ROOT/'SOURCE_MANIFEST.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2,ensure_ascii=False)
    public=files+[ROOT/'SOURCE_MANIFEST.json']
    largest=max(public,key=lambda f:f.stat().st_size)
    report=dict(public_files_excluding_this_audit=len(public),public_bytes_excluding_this_audit=sum(f.stat().st_size for f in public),
        local_total_bytes_including_ignored=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file()),
        largest_public_file=largest.relative_to(ROOT).as_posix(),largest_public_file_bytes=largest.stat().st_size,
        parsed_python_files=py,checked_local_markdown_links=links,broken_links=errors,
        runtime_code_absolute_old_drive_paths=drive,credential_pattern_candidate_files=credentials,
        scope='Static/package integrity only. Does not rerun training, SAM, geometry or the browser.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if errors or drive or credentials:raise SystemExit(2)


if __name__=='__main__':main()
