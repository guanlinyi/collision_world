"""Check the Git publication set and create new inventory/audit files.

Does not stage, overwrite, commit, push or modify source/data files.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
SELF_METADATA = {
    'SOURCE_MANIFEST.json',
    'docs/validation/output_json/package_audit_final.json',
    'docs/publication/output_json/release_check.json',
    'docs/publication/output_json/release_check_final.json',
}
SECRET = re.compile(r'gh[opusr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
LINK = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')

def save_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    metadata_paths = SELF_METADATA | {args.report.resolve().relative_to(ROOT).as_posix()}
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    paths = sorted(set(item.decode('utf-8') for item in raw.split(b'\0') if item))
    path_set = set(paths)
    forbidden, broken, credentials, syntax_errors = [], [], [], []
    files, python_count, link_count = [], 0, 0
    for relative in paths:
        path = ROOT / relative
        if ('experiments' in path.relative_to(ROOT).parts or '.git' in path.relative_to(ROOT).parts
                or any(part.startswith(('browser_profile', 'browser_state')) for part in path.relative_to(ROOT).parts)
                or path.name.endswith('_20261115.md')):
            forbidden.append(relative)
        content = path.read_bytes()
        if path.suffix.lower() in {'.py','.js','.mjs','.cjs','.json','.md','.txt','.yaml','.yml','.toml','.sh','.ps1'}:
            text = content.decode('utf-8-sig')
            if SECRET.search(text):
                credentials.append(relative)
            if path.suffix.lower() == '.py':
                python_count += 1
                try:
                    ast.parse(text, filename=relative)
                except SyntaxError as exc:
                    syntax_errors.append({'path':relative, 'line':exc.lineno, 'message':exc.msg})
            if path.suffix.lower() == '.md':
                for match in LINK.finditer(text):
                    target = match.group(1).strip().strip('<>')
                    if re.match(r'^(?:https?://|mailto:|#)', target):
                        continue
                    target = unquote(target.split('#',1)[0].split('?',1)[0])
                    if not target:
                        continue
                    link_count += 1
                    resolved = (path.parent / target).resolve()
                    if not resolved.is_relative_to(ROOT):
                        broken.append({'from':relative,'target':target,'reason':'outside repository'})
                        continue
                    target_relative = resolved.relative_to(ROOT).as_posix()
                    if target_relative not in path_set and not any(p.startswith(target_relative.rstrip('/') + '/') for p in paths):
                        broken.append({'from':relative,'target':target,'reason':'not in Git publication set'})
        if relative not in metadata_paths:
            files.append({'path':relative,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    previous = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    previous.update(status='Research release source inventory; upstream attribution and validation boundaries retained',
        hash_authority='files is the release-time hash authority. Supplemental transfer manifests retain historical copy-time hashes.',
        exclusions='Git-ignored local journals, experiments, browser data and bytecode; root manifest and audit metadata excluded from recursive hashes.',
        files=files)
    save_new(args.inventory, previous)
    report = dict(public_files=len(paths), inventoried_files=len(files), total_bytes=sum(f['bytes'] for f in files),
        rgb_bytes=sum(f['bytes'] for f in files if '/input_png/rgb/' in f['path']),
        python_files_parsed=python_count, local_markdown_links_checked=link_count,
        forbidden_paths=forbidden, broken_public_links=broken, credential_candidates=credentials, syntax_errors=syntax_errors,
        largest_file=max(files,key=lambda item:item['bytes']),
        passed=not any((forbidden,broken,credentials,syntax_errors)))
    save_new(args.report, report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
