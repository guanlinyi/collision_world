"""Check a generated offline walkthrough without starting models or simulation."""
from __future__ import annotations
import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]

class StoryParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.assets = []
        self.links = []
        self.inline_scripts = []
        self.script = None
        self.headings = []
        self.heading = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        if tag in {'script','img','iframe','source'} and attrs.get('src'):
            self.assets.append(attrs['src'])
        if tag == 'link' and attrs.get('href'):
            self.assets.append(attrs['href'])
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
        if tag == 'script' and not attrs.get('src') and attrs.get('type','') in {'','text/javascript','application/javascript','module'}:
            self.script = []
        if tag in {'h1','h2','h3'}:
            self.heading = []

    def handle_data(self, value):
        if self.script is not None:
            self.script.append(value)
        if self.heading is not None:
            self.heading.append(value)

    def handle_endtag(self, tag):
        if tag == 'script' and self.script is not None:
            self.inline_scripts.append(''.join(self.script))
            self.script = None
        if tag in {'h1','h2','h3'} and self.heading is not None:
            self.headings.append(''.join(self.heading))
            self.heading = None

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--html', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--script-directory', type=Path)
    args = cli.parse_args()
    page = args.html.resolve()
    if not page.is_relative_to(ROOT) or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Inputs and outputs must stay in this project')
    text = page.read_text(encoding='utf-8')
    parsed = StoryParser()
    parsed.feed(text)
    external_assets = [url for url in parsed.assets if urlsplit(url).scheme in {'http','https'}]
    missing = []
    for url in parsed.assets + parsed.links:
        target = urlsplit(url)
        if target.scheme or not target.path:
            continue
        local = (page.parent / unquote(target.path)).resolve()
        if not local.is_relative_to(ROOT) or not local.exists():
            missing.append(url)
    duplicates = [key for key, count in Counter(parsed.ids).items() if count > 1]
    script_files = []
    if args.script_directory:
        scripts = args.script_directory.resolve()
        if not scripts.is_relative_to(ROOT):
            raise ValueError('Script extraction outside project')
        scripts.mkdir(parents=True, exist_ok=False)
        for index, script in enumerate(parsed.inline_scripts):
            file = scripts / f'inline_{index}.js'
            with file.open('x', encoding='utf-8') as handle:
                handle.write(script)
            script_files.append(str(file.relative_to(ROOT)))
    report = dict(html=str(page.relative_to(ROOT)), bytes=page.stat().st_size,
        duplicate_static_ids=duplicates, automatic_external_assets=external_assets,
        missing_local_targets=missing, extracted_scripts=script_files,
        invalid_unicode_replacement='\ufffd' in text,
        passed=not any((duplicates,external_assets,missing,'\ufffd' in text)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
