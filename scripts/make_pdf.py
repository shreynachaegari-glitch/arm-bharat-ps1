"""Build the submission PDF from docs/TECHNICAL_REPORT.md.

Markdown -> HTML -> PDF through headless Chromium. The markdown package is fetched into uv's own cache, so the
pinned project environment is not touched. Usage:
  CHROME=<chromium binary> uv run --no-project --python 3.12 --with markdown==3.7 python scripts/make_pdf.py [out.pdf]
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parents[1] / "docs"
SOURCE = DOCS / "TECHNICAL_REPORT.md"
CHROME_TIMEOUT_S = 120
CSS = """
@page { size: A4; margin: 17mm 16mm 18mm; @bottom-center { content: counter(page) " / " counter(pages);
        font: 8pt "Liberation Sans", sans-serif; color: #666; } }
body { font: 10pt/1.42 "Liberation Sans", Arial, sans-serif; color: #1a1a1a; margin: 0; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 6pt; color: #14532d; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-bottom: 2pt; border-bottom: 1.2pt solid #14532d; color: #14532d; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
h2, h3 { break-after: avoid; }
p, ul, ol { margin: 0 0 6pt; } li { margin: 0 0 2pt; } ul, ol { padding-left: 16pt; }
blockquote { margin: 8pt 0; padding: 7pt 10pt; background: #f0f6f1; border-left: 3pt solid #14532d; }
blockquote p { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font-size: 8.6pt; line-height: 1.3;
        break-inside: avoid; }
th, td { border: .5pt solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; overflow-wrap: break-word; }
th { background: #e6efe8; } tr { break-inside: avoid; } thead { display: table-header-group; }
td code { overflow-wrap: anywhere; }
table:has(img) { table-layout: fixed; }
td img { display: block; width: 100%; height: 98mm; object-fit: cover; object-position: top; }
td img[src*="01_home"] { object-fit: contain; } td img[src*="_leaf_spot"] { object-position: 50% 64%; } td img[src*="no_match"] { object-position: bottom; }
code { font: 8.6pt "DejaVu Sans Mono", monospace; overflow-wrap: anywhere; }
pre { background: #f6f6f6; border: .5pt solid #ddd; padding: 7pt; break-inside: avoid; }
pre code { font-size: 7.1pt; line-height: 1.25; white-space: pre; }
hr { border: 0; margin: 4pt 0; } a { color: #14532d; }
"""


def separate_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list and four spaces per nesting level; GitHub does not."""
    out: list[str] = []
    for line in text.splitlines():
        item = re.match(r"( *)(- |\d+\. )", line)
        previous = out[-1] if out else ""
        if item and not item[1] and previous and not re.match(r" *(- |\d+\. )", previous):
            out.append("")
        if item and item[1]:
            line = "    " + line.lstrip()
        out.append(line)
    return "\n".join(out) + "\n"


def to_html(text: str) -> str:
    text = re.sub(r"(?<![<(\[`\w/])(https://\S+)", r"<\1>", text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><base href="{DOCS.as_uri()}/">'
            f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else DOCS / "PS1_Phase1_Report.pdf").resolve()
    chrome = os.environ.get("CHROME") or sys.exit("set CHROME to a Chromium binary")
    text = separate_lists(SOURCE.read_text())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(to_html(text))
        out.unlink(missing_ok=True)
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-first-run", "--allow-file-access-from-files",
                        "--no-pdf-header-footer", f"--user-data-dir={tmp}/profile", f"--print-to-pdf={out}",
                        page.as_uri()], check=True, timeout=CHROME_TIMEOUT_S, capture_output=True)
    if not out.exists():
        sys.exit("Chromium wrote no PDF")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
