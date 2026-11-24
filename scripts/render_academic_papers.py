#!/usr/bin/env python3
"""Render US-Letter single-spaced IEEE-style academic PDFs (EN & ZH) via Headless Chrome.

Converts ``docs/EXPERIMENT_REPORT.md`` and ``docs/EXPERIMENT_REPORT_ZH.md`` into
publication-grade US-Letter (8.5 x 11 in) single-spaced PDFs with:
  - Strict CSS Paged Media (``@page { size: letter; margin: 0.72in 0.72in 0.78in 0.72in; }``)
  - Crisp serif typography (Times/STIX/Georgia EN; Times + Songti SC / STSong ZH)
  - Boxed Abstract + Index Terms / 关键词 header
  - Booktabs-styled tables (zero horizontal overflow)
  - Inline embedded vector SVG figures with IEEE captions
  - Clean semantic HTML/CSS math rendering (zero raw ``$...$`` or LaTeX macro leaks)
  - Hanging-indent IEEE References / 参考文献
"""
from __future__ import annotations

import argparse
import html as _html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

DEFAULT_CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ---------------------------------------------------------------------------
# Recursive-Descent LaTeX-to-HTML Math Renderer
# ---------------------------------------------------------------------------
_MATH_SYMBOLS: dict[str, str] = {
    "alpha": "&alpha;",
    "beta": "&beta;",
    "gamma": "&gamma;",
    "delta": "&delta;",
    "Delta": "&Delta;",
    "epsilon": "&epsilon;",
    "varepsilon": "&varepsilon;",
    "theta": "&theta;",
    "lambda": "&lambda;",
    "mu": "&mu;",
    "pi": "&pi;",
    "rho": "&rho;",
    "sigma": "&sigma;",
    "tau": "&tau;",
    "phi": "&phi;",
    "omega": "&omega;",
    "Omega": "&Omega;",
    "odot": "&#x2299;",
    "otimes": "&#x2297;",
    "oplus": "&#x2295;",
    "top": "&#x22A4;",
    "cdot": "&middot;",
    "times": "&times;",
    "approx": "&approx;",
    "sim": "&sim;",
    "le": "&le;",
    "leq": "&le;",
    "ge": "&ge;",
    "geq": "&ge;",
    "ne": "&ne;",
    "neq": "&ne;",
    "in": "&isin;",
    "notin": "&notin;",
    "to": "&rarr;",
    "rightarrow": "&rarr;",
    "leftarrow": "&larr;",
    "Rightarrow": "&rArr;",
    "pm": "&plusmn;",
    "infty": "&infin;",
    "partial": "&part;",
    "nabla": "&nabla;",
    "langle": "&#x27E8;",
    "rangle": "&#x27E9;",
    "lvert": "|",
    "rvert": "|",
    "lVert": "&#x2016;",
    "rVert": "&#x2016;",
    "mid": "|",
    "quad": "<span class='m-quad'></span>",
    "qquad": "<span class='m-qquad'></span>",
    ",": "<span class='m-thin'></span>",
    ";": "<span class='m-med'></span>",
    ":": "<span class='m-med'></span>",
    "!": "",
    " ": "&nbsp;",
}

_MATH_OPERATORS: set[str] = {
    "max", "min", "tanh", "sinh", "cosh", "exp", "log", "ln", "sin", "cos",
    "arg", "argmax", "argmin", "softmax", "sigmoid", "diag", "Diag", "det", "tr",
}


def _read_balanced_group(src: str, pos: int) -> tuple[str, int]:
    """Read a ``{...}`` group or single token starting at ``pos``."""
    n = len(src)
    while pos < n and src[pos].isspace():
        pos += 1
    if pos >= n:
        return "", pos
    if src[pos] == "{":
        depth = 0
        start = pos + 1
        i = pos
        while i < n:
            ch = src[i]
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return src[start:i], i + 1
            i += 1
        return src[start:], n
    if src[pos] == "\\":
        j = pos + 1
        if j < n and src[j].isalpha():
            while j < n and src[j].isalpha():
                j += 1
            return src[pos:j], j
        return src[pos : min(pos + 2, n)], min(pos + 2, n)
    return src[pos], pos + 1


def _render_math_expr(src: str, *, bold: bool = False, roman: bool = False) -> str:
    """Convert a LaTeX math expression into semantic HTML without raw LaTeX escapes."""
    out: list[str] = []
    i = 0
    n = len(src)

    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue

        # Handle subscripts and superscripts attached to previous token
        if ch in ("_", "^"):
            sub_html: str | None = None
            sup_html: str | None = None
            while i < n and src[i] in ("_", "^"):
                kind = src[i]
                grp, i = _read_balanced_group(src, i + 1)
                rendered = _render_math_expr(grp, bold=False, roman=False)
                if kind == "_":
                    sub_html = rendered
                else:
                    sup_html = rendered
                while i < n and src[i].isspace():
                    i += 1
            if sub_html is not None and sup_html is not None:
                out.append(
                    f"<span class='m-subsup'><sup>{sup_html}</sup><sub>{sub_html}</sub></span>"
                )
            elif sub_html is not None:
                out.append(f"<sub>{sub_html}</sub>")
            elif sup_html is not None:
                out.append(f"<sup>{sup_html}</sup>")
            continue

        if ch == "{":
            grp, i = _read_balanced_group(src, i)
            out.append(_render_math_expr(grp, bold=bold, roman=roman))
            continue

        if ch == "\\":
            j = i + 1
            if j >= n:
                break
            if src[j].isalpha():
                while j < n and src[j].isalpha():
                    j += 1
                cmd = src[i + 1 : j]
                i = j
            else:
                cmd = src[j]
                i = j + 1

            if cmd in ("left", "right", "big", "Big", "bigg", "Bigg"):
                while i < n and src[i].isspace():
                    i += 1
                if i < n:
                    if src[i] == ".":
                        i += 1
                    elif src[i] == "\\":
                        tok, i = _read_balanced_group(src, i)
                        sym = _MATH_SYMBOLS.get(tok.lstrip("\\"), _html.escape(tok.lstrip("\\")))
                        out.append(f"<span class='m-delim'>{sym}</span>")
                    else:
                        out.append(f"<span class='m-delim'>{_html.escape(src[i])}</span>")
                        i += 1
                continue

            if cmd in ("mathbf", "boldsymbol", "bm"):
                grp, i = _read_balanced_group(src, i)
                inner = _render_math_expr(grp, bold=True, roman=True)
                out.append(f"<span class='m-bf'>{inner}</span>")
                continue

            if cmd in ("text", "mathrm", "textrm", "operatorname", "mathsf", "mathtt"):
                grp, i = _read_balanced_group(src, i)
                if cmd == "operatorname":
                    out.append(f"<span class='m-op'>{_html.escape(grp)}</span>")
                else:
                    # Render raw text verbatim (preserving dashes/spaces)
                    out.append(f"<span class='m-rm'>{_html.escape(grp)}</span>")
                continue

            if cmd == "hat":
                grp, i = _read_balanced_group(src, i)
                inner = _render_math_expr(grp, bold=bold, roman=roman)
                out.append(f"<span class='m-hat'>{inner}</span>")
                continue

            if cmd == "tilde":
                grp, i = _read_balanced_group(src, i)
                inner = _render_math_expr(grp, bold=bold, roman=roman)
                out.append(f"<span class='m-tilde'>{inner}</span>")
                continue

            if cmd == "bar":
                grp, i = _read_balanced_group(src, i)
                inner = _render_math_expr(grp, bold=bold, roman=roman)
                out.append(f"<span class='m-bar'>{inner}</span>")
                continue

            if cmd == "frac":
                num, i = _read_balanced_group(src, i)
                den, i = _read_balanced_group(src, i)
                num_h = _render_math_expr(num)
                den_h = _render_math_expr(den)
                out.append(
                    f"<span class='m-frac'><span class='m-num'>{num_h}</span>"
                    f"<span class='m-den'>{den_h}</span></span>"
                )
                continue

            if cmd == "sqrt":
                grp, i = _read_balanced_group(src, i)
                inner = _render_math_expr(grp)
                out.append(f"<span class='m-sqrt'>&radic;<span class='m-rad'>{inner}</span></span>")
                continue

            if cmd in _MATH_OPERATORS:
                out.append(f"<span class='m-op'>{cmd}</span>")
                continue

            if cmd in _MATH_SYMBOLS:
                sym = _MATH_SYMBOLS[cmd]
                if cmd in (
                    "cdot", "times", "odot", "otimes", "oplus", "approx", "sim",
                    "le", "leq", "ge", "geq", "ne", "neq", "in", "notin", "to",
                    "rightarrow", "leftarrow", "Rightarrow", "pm",
                ):
                    out.append(f"<span class='m-bin'>{sym}</span>")
                else:
                    out.append(sym)
                continue

            if cmd in ("{", "}", "%", "_", "&", "#"):
                out.append(_html.escape(cmd))
                continue

            # Unknown command fallback: raise explicitly so no LaTeX leak goes unnoticed!
            raise ValueError(f"Unsupported LaTeX math command '\\{cmd}' in expression: {src!r}")

        # Binary operators and relations
        if ch in ("=", "+", "<", ">"):
            out.append(f"<span class='m-bin'>{_html.escape(ch)}</span>")
            i += 1
            continue
        if ch == "-":
            # Unary minus if at start or right after '(' / ',' / '=' / '+' / '^' / '_'
            prev = out[-1] if out else ""
            if not prev or prev.endswith(("(", ",", "m-bin'>", "m-delim'>")):
                out.append("&minus;")
            else:
                out.append("<span class='m-bin'>&minus;</span>")
            i += 1
            continue
        if ch == ",":
            out.append(",<span class='m-thin'></span>")
            i += 1
            continue
        if ch in ("(", ")", "[", "]", "|", "/", ".", ":", ";"):
            out.append(_html.escape(ch))
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < n and (src[j].isdigit() or (src[j] == "." and j + 1 < n and src[j + 1].isdigit())):
                j += 1
            out.append(f"<span class='m-numlit'>{src[i:j]}</span>")
            i = j
            continue
        if ch.isalpha():
            if bold:
                out.append(f"<span class='m-bf'>{_html.escape(ch)}</span>")
            elif roman:
                out.append(f"<span class='m-rm'>{_html.escape(ch)}</span>")
            else:
                out.append(f"<i>{_html.escape(ch)}</i>")
            i += 1
            continue

        out.append(_html.escape(ch))
        i += 1

    return "".join(out)


def render_latex_math(expr: str, *, display: bool = False) -> str:
    """Render a LaTeX math string to clean HTML."""
    body = _render_math_expr(expr.strip())
    if display:
        return f"<div class='math-display'><span class='math'>{body}</span></div>"
    return f"<span class='math'>{body}</span>"


def _protect_code_and_render_math(md_text: str) -> tuple[str, dict[str, str]]:
    """Replace display & inline math with placeholders before Markdown parsing (ignoring code spans/fences)."""
    placeholders: dict[str, str] = {}
    counter = [0]

    def _stash(html_snippet: str, *, block: bool = False) -> str:
        idx = counter[0]
        counter[0] += 1
        key = f"XMATHBLK{idx}X" if block else f"XMATHINL{idx}X"
        placeholders[key] = html_snippet
        return f"\n\n{key}\n\n" if block else key

    # Split by fenced code blocks first so we never touch code fences
    fence_re = re.compile(r"(```.*?```)", re.S)
    parts = fence_re.split(md_text)
    out_parts: list[str] = []

    inline_code_re = re.compile(r"(`+[^`\n]+?`+)")
    display_re = re.compile(r"\$\$(.*?)\$\$", re.S)
    inline_re = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)")

    for idx_p, part in enumerate(parts):
        if idx_p % 2 == 1:
            out_parts.append(part)
            continue

        # Protect inline code spans before matching $...$
        code_tokens: list[str] = []

        def _save_code(m: re.Match[str], tokens: list[str] = code_tokens) -> str:
            ci = len(tokens)
            tokens.append(m.group(0))
            return f"XCODETOK{ci}X"

        part_nc = inline_code_re.sub(_save_code, part)

        # 1. Display math $$...$$
        part_nc = display_re.sub(
            lambda m: _stash(render_latex_math(m.group(1), display=True), block=True),
            part_nc,
        )
        # 2. Inline math $...$
        part_nc = inline_re.sub(
            lambda m: _stash(render_latex_math(m.group(1), display=False), block=False),
            part_nc,
        )

        # Restore inline code spans
        for ci, tok in enumerate(code_tokens):
            part_nc = part_nc.replace(f"XCODETOK{ci}X", tok)
        out_parts.append(part_nc)

    return "".join(out_parts), placeholders


# ---------------------------------------------------------------------------
# Leak Detector (asserts zero raw LaTeX / unrendered $...$ in final HTML)
# ---------------------------------------------------------------------------
_LEAK_LATEX_RE = re.compile(
    r"\\(?:mathbf|mathrm|operatorname|hat|tilde|bar|odot|top|langle|rangle|lvert|rvert|"
    r"tanh|text|approx|le|leq|ge|geq|cdot|times|lambda|gamma|alpha|beta|delta|Delta|"
    r"left|right|frac|sqrt|quad|qquad)\b"
)
_LEAK_DOLLAR_RE = re.compile(r"(?<!\w)\$(?!\d)([^$<>\n]{1,120})\$")


def find_unrendered_math_leaks(html_text: str) -> list[str]:
    """Return a list of any unrendered LaTeX macros or raw ``$...$`` math spans in HTML."""
    # Strip <style>, <svg>, <pre>, and <code> blocks before checking visible prose/tables
    clean = re.sub(r"<style\b[^>]*>.*?</style>", "", html_text, flags=re.S | re.I)
    clean = re.sub(r"<svg\b[^>]*>.*?</svg>", "", clean, flags=re.S | re.I)
    clean = re.sub(r"<pre\b[^>]*>.*?</pre>", "", clean, flags=re.S | re.I)
    clean = re.sub(r"<code\b[^>]*>.*?</code>", "", clean, flags=re.S | re.I)
    text_only = re.sub(r"<[^>]+>", " ", clean)
    leaks: list[str] = []
    for m in _LEAK_LATEX_RE.finditer(text_only):
        leaks.append(f"LaTeX macro leak: {m.group(0)!r}")
    for m in _LEAK_DOLLAR_RE.finditer(text_only):
        leaks.append(f"Raw dollar math leak: {m.group(0)!r}")
    return leaks


# ---------------------------------------------------------------------------
# Inline Vector SVG Embedding + Academic Structural Post-Processing
# ---------------------------------------------------------------------------
def _embed_svg_figures(body_html: str, base_dir: Path) -> str:
    """Convert ``<p><img src="...svg"></p>`` + optional ``<p><em>Fig...</em></p>`` into ``<figure>``."""
    fig_with_cap_re = re.compile(
        r'<p>\s*<img\s+src="([^"]+\.svg)"\s+alt="([^"]*)"\s*/?>\s*</p>\s*'
        r'<p>\s*<em>(.*?)</em>\s*</p>',
        re.S,
    )
    fig_no_cap_re = re.compile(
        r'<p>\s*<img\s+src="([^"]+\.svg)"\s+alt="([^"]*)"\s*/?>\s*</p>',
        re.S,
    )

    def _load_inline_svg(src_rel: str) -> str:
        svg_path = (base_dir / src_rel).resolve()
        if not svg_path.is_file():
            raise FileNotFoundError(f"Referenced SVG figure not found: {svg_path}")
        raw = svg_path.read_text(encoding="utf-8").strip()
        # Strip XML prolog if any
        raw = re.sub(r"^<\?xml[^>]*\?>\s*", "", raw)
        # Ensure responsive vector scaling in CSS print layout
        m_open = re.match(r"<svg\b([^>]*)>", raw)
        if m_open:
            attrs = m_open.group(1)
            # Extract original width/height to preserve viewBox if missing
            mw = re.search(r'\bwidth="(\d+(?:\.\d+)?)"', attrs)
            mh = re.search(r'\bheight="(\d+(?:\.\d+)?)"', attrs)
            if "viewBox=" not in attrs and mw and mh:
                attrs += f' viewBox="0 0 {mw.group(1)} {mh.group(1)}"'
            attrs = re.sub(r'\s*\bwidth="[^"]*"', "", attrs)
            attrs = re.sub(r'\s*\bheight="[^"]*"', "", attrs)
            raw = f'<svg{attrs} width="100%" preserveAspectRatio="xMidYMid meet">' + raw[m_open.end() :]
        return raw

    def _repl_with_cap(m: re.Match[str]) -> str:
        svg_inline = _load_inline_svg(m.group(1))
        cap = m.group(3).strip()
        return (
            f'<figure class="fig">\n'
            f'  <div class="fig-svg">{svg_inline}</div>\n'
            f'  <figcaption>{cap}</figcaption>\n'
            f'</figure>'
        )

    def _repl_no_cap(m: re.Match[str]) -> str:
        svg_inline = _load_inline_svg(m.group(1))
        alt = m.group(2).strip()
        cap_html = f"  <figcaption>{alt}</figcaption>\n" if alt else ""
        return f'<figure class="fig">\n  <div class="fig-svg">{svg_inline}</div>\n{cap_html}</figure>'

    body_html = fig_with_cap_re.sub(_repl_with_cap, body_html)
    body_html = fig_no_cap_re.sub(_repl_no_cap, body_html)
    return body_html


def _style_callouts(body_html: str) -> str:
    """Turn GitHub-style ``> [!IMPORTANT]`` / ``> [!NOTE]`` blockquotes into academic callout boxes."""
    callout_re = re.compile(
        r"<blockquote>\s*<p>\[!(IMPORTANT|NOTE|TIP|WARNING)\]\s*(?:<br\s*/?>|\n)?(.*?)</p>\s*</blockquote>",
        re.S,
    )

    def _repl(m: re.Match[str]) -> str:
        kind = m.group(1).lower()
        inner = m.group(2).strip()
        return f'<div class="callout callout-{kind}">{inner}</div>'

    return callout_re.sub(_repl, body_html)


def _style_header_and_abstract(body_html: str, lang: str) -> str:
    """Wrap title+metadata into ``<header class="paper-header">`` and Abstract+Index Terms in ``.abstract-box``."""
    # 1. Title <h1> + immediate metadata <p>
    hdr_re = re.compile(r"^\s*(<h1>.*?</h1>)\s*(<p>.*?</p>)", re.S)
    body_html = hdr_re.sub(
        r'<header class="paper-header">\1\n<div class="paper-meta">\2</div>\n</header>',
        body_html,
        count=1,
    )

    # 2. Abstract + Index Terms (EN or ZH)
    abs_heading = "摘要" if lang == "zh" else "Abstract"
    abs_re = re.compile(
        rf"<h2>{re.escape(abs_heading)}</h2>\s*(<p>.*?</p>)\s*(<p><strong>(?:Index Terms|关键词).*?</p>)",
        re.S,
    )
    abs_repl = (
        f'<section class="abstract-box"><div class="abstract-title">{abs_heading}</div>'
        r'\1\n<div class="index-terms">\2</div></section>'
    )
    body_html = abs_re.sub(abs_repl, body_html, count=1)

    # Remove redundant top horizontal rule right before Section I
    body_html = re.sub(
        r"</section>\s*(<div class=\"callout.*?</div>)?\s*<hr\s*/?>",
        r"</section>\n\1",
        body_html,
        count=1,
        flags=re.S,
    )
    return body_html


def _insert_code_wbr(body_html: str) -> str:
    """Insert soft ``<wbr>`` breakpoints after ``_`` and ``/`` in long inline ``<code>`` spans (never in ``<pre>``)."""
    # Protect <pre>...</pre> blocks
    pre_blocks: list[str] = []

    def _save_pre(m: re.Match[str]) -> str:
        idx = len(pre_blocks)
        pre_blocks.append(m.group(0))
        return f"XPREBLK{idx}X"

    work = re.sub(r"<pre\b[^>]*>.*?</pre>", _save_pre, body_html, flags=re.S)

    def _add_wbr(m: re.Match[str]) -> str:
        inner = m.group(1)
        plain = re.sub(r"<[^>]+>", "", inner)
        if len(plain) < 15:
            return m.group(0)
        broken = re.sub(r"([_/])", r"\1<wbr>", inner)
        return f"<code>{broken}</code>"

    work = re.sub(r"<code>(.*?)</code>", _add_wbr, work, flags=re.S)
    for idx, blk in enumerate(pre_blocks):
        work = work.replace(f"XPREBLK{idx}X", blk)
    return work


def _mark_nowrap_badges(body_html: str) -> str:
    """Prevent short badge-like table cells (e.g. Measured, Formula, stock, 实测) from breaking mid-word."""
    badge_words = {
        "Measured", "Formula", "Derived", "PASS", "FAIL", "stock",
        "fused=True", "fused=False", "True→False",
        "实测", "推导", "通过", "失败", "是/是", "否/否", "是/否", "否/是",
        "N/N", "N/Y", "Y/N", "Y/Y",
    }

    def _repl_td(m: re.Match[str]) -> str:
        attrs = m.group(1)
        inner = m.group(2)
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        if plain in badge_words or (len(plain) <= 10 and " " not in plain and "/" not in plain and "_" not in plain):
            if 'class="' in attrs:
                attrs = attrs.replace('class="', 'class="td-nowrap ')
            else:
                attrs = f'{attrs} class="td-nowrap"'
        return f"<td{attrs}>{inner}</td>"

    return re.sub(r"<td([^>]*)>(.*?)</td>", _repl_td, body_html, flags=re.S)


def _wrap_table_blocks(body_html: str) -> str:
    """Group ``<h4>Table ...</h4><table>...</table>`` and mark compact tables ``break-inside: avoid``."""
    tbl_re = re.compile(r"(<h4>.*?</h4>)\s*(<table>.*?</table>)", re.S)

    def _repl(m: re.Match[str]) -> str:
        cap, tbl = m.group(1), m.group(2)
        n_rows = len(re.findall(r"<tr\b", tbl))
        cls = "table-block table-compact" if n_rows <= 15 else "table-block table-long"
        return f'<div class="{cls}">\n{cap}\n{tbl}\n</div>'

    return tbl_re.sub(_repl, body_html)


def _style_references_section(body_html: str, lang: str) -> str:
    """Wrap the References / 参考文献 section in a crisp two-column flex IEEE reference list."""
    ref_heading = "参考文献" if lang == "zh" else "References"
    marker = f"<h2>{ref_heading}</h2>"
    idx = body_html.find(marker)
    if idx < 0:
        return body_html
    prefix = body_html[:idx]
    suffix = body_html[idx + len(marker) :]
    tail_m = re.search(r"(<hr\s*/?>|<h[23]>)", suffix)
    if tail_m:
        refs_raw = suffix[: tail_m.start()]
        tail = suffix[tail_m.start() :]
    else:
        refs_raw = suffix
        tail = ""

    # Turn each <p>[k] ...</p> into a flexbox item so [k] is never clipped by text-indent inheritance
    ref_item_re = re.compile(r"<p>(\[\d+\])\s*(.*?)</p>", re.S)
    refs_styled = ref_item_re.sub(
        r'<div class="ref-item"><span class="ref-num">\1</span><span class="ref-body">\2</span></div>',
        refs_raw,
    )
    return f'{prefix}<section class="references">\n{marker}\n{refs_styled}\n</section>\n{tail}'


# ---------------------------------------------------------------------------
# Strict US-Letter Single-Spaced Academic CSS
# ---------------------------------------------------------------------------
def _build_css(lang: str) -> str:
    is_zh = lang == "zh"
    body_font = (
        '"Times New Roman", "Songti SC", "STSong", "Noto Serif CJK SC", "PingFang SC", serif'
        if is_zh
        else '"Times New Roman", "STIX Two Text", "Georgia", serif'
    )
    heading_font = (
        '"Times New Roman", "PingFang SC", "Heiti SC", "Songti SC", sans-serif'
        if is_zh
        else '"Times New Roman", "STIX Two Text", "Georgia", serif'
    )
    body_size = "9.8pt" if is_zh else "10pt"
    body_lh = "1.30" if is_zh else "1.18"
    table_size = "7.9pt" if is_zh else "7.8pt"
    table_lh = "1.22" if is_zh else "1.18"
    ref_size = "8.1pt"

    return f"""
@page {{
  size: letter;
  margin: 0.72in 0.72in 0.78in 0.72in;
  @bottom-center {{
    content: counter(page);
    font-family: "Times New Roman", "STIX Two Text", Georgia, serif;
    font-size: 9pt;
    color: #333;
  }}
}}

* {{
  box-sizing: border-box;
}}

html, body {{
  margin: 0;
  padding: 0;
  background: #ffffff;
  color: #111111;
  font-family: {body_font};
  font-size: {body_size};
  line-height: {body_lh};
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}

main.paper {{
  width: 100%;
}}

/* Title & Metadata Header */
.paper-header {{
  text-align: center;
  padding-bottom: 7pt;
  margin-bottom: 8pt;
  border-bottom: 1.2pt solid #1a2433;
}}

.paper-header h1 {{
  font-family: {heading_font};
  font-size: {"14.2pt" if is_zh else "14.8pt"};
  font-weight: 700;
  line-height: 1.24;
  margin: 0 0 6pt 0;
  color: #0f172a;
}}

.paper-meta p {{
  margin: 0;
  font-size: 8.3pt;
  line-height: 1.32;
  color: #2d3748;
  text-align: center;
}}

.paper-meta a {{
  color: #1a365d;
  text-decoration: none;
}}

/* Abstract & Index Terms Box */
.abstract-box {{
  background: #f7f9fc;
  border: 0.6pt solid #cfd8e3;
  border-left: 2.8pt solid #1a365d;
  padding: 7.5pt 10pt 7pt 10pt;
  margin: 0 0 8pt 0;
  break-inside: avoid;
  page-break-inside: avoid;
}}

.abstract-title {{
  font-family: {heading_font};
  font-size: 9.4pt;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: {"0.06em" if is_zh else "0.05em"};
  color: #1a365d;
  margin: 0 0 3.5pt 0;
}}

.abstract-box p {{
  margin: 0 0 4.5pt 0;
  font-size: {"9.1pt" if is_zh else "9.2pt"};
  line-height: {"1.28" if is_zh else "1.18"};
  text-align: justify;
}}

.abstract-box .index-terms p {{
  margin: 0;
  font-size: 8.6pt;
  color: #1e293b;
}}

/* Callout Box */
.callout {{
  background: #f8fafc;
  border: 0.5pt solid #d8dee6;
  border-left: 2.5pt solid #2b6cb0;
  padding: 5.5pt 8.5pt;
  margin: 5pt 0 8pt 0;
  font-size: 8.4pt;
  line-height: {"1.26" if is_zh else "1.18"};
  color: #1e293b;
  text-align: left;
  break-inside: avoid;
  page-break-inside: avoid;
}}

/* Headings */
h2 {{
  font-family: {heading_font};
  font-size: 11pt;
  font-weight: 700;
  color: #0f172a;
  margin: 10.5pt 0 4pt 0;
  padding-bottom: 2pt;
  border-bottom: 0.65pt solid #475569;
  letter-spacing: {"0.02em" if is_zh else "0.025em"};
  break-after: avoid;
  page-break-after: avoid;
}}

h3 {{
  font-family: {heading_font};
  font-size: 10pt;
  font-weight: 700;
  font-style: {"normal" if is_zh else "italic"};
  color: #1e293b;
  margin: 8.5pt 0 3pt 0;
  break-after: avoid;
  page-break-after: avoid;
}}

h4 {{
  font-family: {heading_font};
  font-size: 8.6pt;
  font-weight: 700;
  color: #0f172a;
  margin: 7.5pt 0 2.5pt 0;
  line-height: 1.22;
  break-after: avoid;
  page-break-after: avoid;
}}

/* Body Paragraphs & Lists */
p {{
  margin: 0 0 5pt 0;
  text-align: justify;
  orphans: 3;
  widows: 3;
}}

ul, ol {{
  margin: 2pt 0 6pt 0;
  padding-left: 16pt;
}}

li {{
  margin: 0 0 2.8pt 0;
  text-align: justify;
}}

hr {{
  border: none;
  border-top: 0.6pt solid #cbd5e1;
  margin: 8pt 0;
}}

a {{
  color: #1a365d;
  text-decoration: none;
}}

/* Inline Code & Preformatted Blocks */
code {{
  font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
  font-size: 8.0pt;
  background: #f3f5f8;
  color: #1a2433;
  padding: 0.4pt 2pt;
  border-radius: 1.8pt;
  word-break: normal;
  overflow-wrap: break-word;
}}

pre {{
  background: #f6f8fa;
  border: 0.5pt solid #d8dee6;
  border-radius: 2.5pt;
  padding: 5pt 7.5pt;
  margin: 4.5pt 0 7pt 0;
  overflow-x: hidden;
  break-inside: avoid;
  page-break-inside: avoid;
}}

pre code {{
  background: transparent;
  padding: 0;
  font-size: 7.6pt;
  line-height: 1.26;
  white-space: pre-wrap;
  word-break: break-all;
}}

/* Booktabs Academic Tables */
.table-block {{
  margin: 5pt 0 7pt 0;
}}

.table-compact {{
  break-inside: avoid;
  page-break-inside: avoid;
}}

.table-block h4 {{
  margin-top: 0;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  border-top: 1.4pt solid #111111;
  border-bottom: 1.4pt solid #111111;
  font-size: {table_size};
  line-height: {table_lh};
  margin: 0;
  table-layout: auto;
}}

thead {{
  display: table-header-group;
  border-bottom: 0.75pt solid #222222;
  background: #f7f8fa;
}}

th {{
  font-weight: 700;
  padding: 3.2pt 4pt;
  color: #0f172a;
  vertical-align: bottom;
}}

td {{
  padding: 2.6pt 4pt;
  vertical-align: top;
  border-bottom: 0.3pt solid #e2e8f0;
  overflow-wrap: break-word;
  word-break: normal;
}}

tbody tr:nth-child(even) {{
  background: #fbfcfd;
}}

tbody tr:last-child td {{
  border-bottom: none;
}}

tr {{
  break-inside: avoid;
  page-break-inside: avoid;
}}

th code, td code {{
  background: transparent;
  padding: 0;
  font-size: 0.91em;
  color: #0f172a;
  word-break: normal;
  overflow-wrap: break-word;
}}

td.td-nowrap {{
  white-space: nowrap;
}}

/* Vector Figures */
figure.fig {{
  margin: 5.5pt 0 7pt 0;
  padding: 0;
  break-inside: avoid;
  page-break-inside: avoid;
  text-align: center;
}}

.fig-svg {{
  width: 100%;
  margin: 0 auto;
}}

.fig-svg svg {{
  width: 100%;
  height: auto;
  max-height: 4.15in;
  display: block;
  margin: 0 auto;
}}

figcaption {{
  font-size: 8.2pt;
  line-height: {"1.26" if is_zh else "1.20"};
  color: #1e293b;
  text-align: justify;
  margin-top: 3.5pt;
  padding: 0 2pt;
}}

figcaption code {{
  font-size: 7.7pt;
}}

/* Semantic Math Typography */
.math-display {{
  text-align: center;
  margin: 6pt 0 7pt 0;
  padding: 4.5pt 6pt;
  background: #fafbfc;
  border-top: 0.4pt solid #e2e8f0;
  border-bottom: 0.4pt solid #e2e8f0;
  font-size: 10.2pt;
  break-inside: avoid;
  page-break-inside: avoid;
}}

.math {{
  font-family: "Times New Roman", "STIX Two Math", "STIX Two Text", Georgia, serif;
  white-space: nowrap;
}}

.math-display .math {{
  white-space: normal;
}}

.math i {{
  font-style: italic;
}}

.m-bf {{
  font-weight: 700;
  font-style: normal;
}}

.m-rm, .m-op, .m-numlit {{
  font-style: normal;
}}

.m-op {{
  margin: 0 1.2pt;
}}

.m-bin {{
  margin: 0 2pt;
}}

.m-delim {{
  font-size: 1.06em;
  vertical-align: -0.02em;
}}

.m-thin {{
  display: inline-block;
  width: 1.8pt;
}}

.m-med {{
  display: inline-block;
  width: 3pt;
}}

.m-quad {{
  display: inline-block;
  width: 10pt;
}}

.m-qquad {{
  display: inline-block;
  width: 18pt;
}}

.math > sub, .math > sup {{
  font-size: 0.72em;
  line-height: 0;
}}

.m-subsup {{
  display: inline-flex;
  flex-direction: column;
  justify-content: center;
  vertical-align: 0.34em;
  line-height: 0.96;
  gap: 0.5pt;
  margin-left: 0.8pt;
  text-align: left;
}}

.m-subsup sup, .m-subsup sub {{
  display: block;
  position: static;
  vertical-align: baseline;
  font-size: 0.68em;
  line-height: 0.96;
}}

.m-hat {{
  position: relative;
  display: inline-block;
}}

.m-hat::before {{
  content: "\\02C6";
  position: absolute;
  top: -0.22em;
  left: 50%;
  transform: translateX(-50%);
  font-style: normal;
  font-weight: 700;
  font-size: 0.88em;
  line-height: 1;
}}

.m-tilde {{
  position: relative;
  display: inline-block;
}}

.m-tilde::before {{
  content: "\\02DC";
  position: absolute;
  top: -0.22em;
  left: 50%;
  transform: translateX(-50%);
  font-style: normal;
  font-size: 0.88em;
  line-height: 1;
}}

.m-bar {{
  border-top: 0.6pt solid currentColor;
  padding-top: 0.5pt;
}}

.m-frac {{
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  vertical-align: -0.35em;
  margin: 0 1.5pt;
  font-size: 0.86em;
}}

.m-num {{
  border-bottom: 0.55pt solid currentColor;
  padding: 0 2pt 0.8pt 2pt;
}}

.m-den {{
  padding: 0.8pt 2pt 0 2pt;
}}

/* IEEE References Section (Flex two-column layout immune to text-indent bugs) */
.references {{
  margin-top: 6pt;
}}

.ref-item {{
  display: flex;
  align-items: baseline;
  font-size: {ref_size};
  line-height: {"1.20" if is_zh else "1.14"};
  margin: 0 0 {"1.9pt" if is_zh else "1.35pt"} 0;
  text-align: left;
  break-inside: avoid;
  page-break-inside: avoid;
}}

.ref-num {{
  flex: 0 0 22pt;
  font-weight: 600;
  color: #0f172a;
}}

.ref-body {{
  flex: 1 1 auto;
  min-width: 0;
}}

.ref-body a {{
  word-break: break-all;
}}
""".strip()


# ---------------------------------------------------------------------------
# End-to-End Markdown -> HTML -> US-Letter PDF Pipeline
# ---------------------------------------------------------------------------
def markdown_to_academic_html(
    md_text: str,
    *,
    base_dir: Path,
    lang: str = "en",
    title: str = "Qwen3.6-35B-A6B-Loop64 Technical Report",
) -> str:
    """Convert an IEEE-formatted Markdown report into a print-ready US-Letter HTML document."""
    protected_md, placeholders = _protect_code_and_render_math(md_text)

    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable(["table", "strikethrough"])
    body_html = md.render(protected_md)

    # Restore math blocks and inline math expressions
    for key, snippet in placeholders.items():
        if key.startswith("XMATHBLK"):
            body_html = body_html.replace(f"<p>{key}</p>", snippet)
        body_html = body_html.replace(key, snippet)

    body_html = _embed_svg_figures(body_html, base_dir=base_dir)
    body_html = _style_callouts(body_html)
    body_html = _style_header_and_abstract(body_html, lang=lang)
    body_html = _insert_code_wbr(body_html)
    body_html = _mark_nowrap_badges(body_html)
    body_html = _wrap_table_blocks(body_html)
    body_html = _style_references_section(body_html, lang=lang)

    html_lang = "zh-Hans" if lang == "zh" else "en"
    css = _build_css(lang)
    full_html = (
        f"<!DOCTYPE html>\n"
        f'<html lang="{html_lang}">\n'
        f"<head>\n"
        f'  <meta charset="utf-8">\n'
        f"  <title>{_html.escape(title)}</title>\n"
        f"  <style>\n{css}\n  </style>\n"
        f"</head>\n"
        f"<body>\n"
        f'  <main class="paper">\n{body_html}\n  </main>\n'
        f"</body>\n"
        f"</html>\n"
    )

    leaks = find_unrendered_math_leaks(full_html)
    if leaks:
        raise ValueError(f"Detected unrendered math/LaTeX leaks in HTML output: {leaks[:8]}")
    return full_html


def get_pdf_page_boxes(pdf_path: Path) -> list[tuple[float, float]]:
    """Parse ``/MediaBox [x0 y0 x1 y1]`` sizes (width_pt, height_pt) directly from PDF bytes."""
    raw = pdf_path.read_bytes()
    # Chrome PDFs emit /MediaBox [0 0 612 792] (or floats) per page object
    matches = re.findall(
        rb"/MediaBox\s*\[\s*([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s*\]",
        raw,
    )
    boxes: list[tuple[float, float]] = []
    for x0, y0, x1, y1 in matches:
        w = float(x1) - float(x0)
        h = float(y1) - float(y0)
        boxes.append((round(w, 2), round(h, 2)))
    return boxes


def _find_chrome_bin(explicit: str | None = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    if os.path.isfile(DEFAULT_CHROME_BIN):
        return DEFAULT_CHROME_BIN
    for cand in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(cand)
        if found:
            return found
    raise FileNotFoundError(
        f"Headless Chrome binary not found at {DEFAULT_CHROME_BIN!r} or in PATH."
    )


def render_pdf_from_markdown(
    md_path: Path,
    pdf_path: Path,
    *,
    lang: str = "en",
    chrome_bin: str | None = None,
    keep_html: Path | None = None,
) -> dict[str, object]:
    """Render a Markdown paper to a US-Letter PDF via headless Chrome and verify MediaBox."""
    md_path = md_path.resolve()
    pdf_path = pdf_path.resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    md_text = md_path.read_text(encoding="utf-8")
    first_h1 = re.search(r"^#\s+(.+)$", md_text, flags=re.M)
    title = first_h1.group(1).strip() if first_h1 else md_path.stem

    full_html = markdown_to_academic_html(
        md_text,
        base_dir=md_path.parent,
        lang=lang,
        title=title,
    )

    chrome = _find_chrome_bin(chrome_bin)
    tmp_dir = Path(tempfile.mkdtemp(prefix="loop64_pdf_"))
    try:
        html_path = keep_html.resolve() if keep_html else (tmp_dir / f"{md_path.stem}.html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(full_html, encoding="utf-8")

        profile_dir = tmp_dir / "chrome_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-sync",
            f"--user-data-dir={profile_dir}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
        # macOS headless Chrome occasionally trips its teardown watchdog *after* writing a valid PDF.
        pdf_ok = pdf_path.is_file() and pdf_path.stat().st_size > 1024 and bool(get_pdf_page_boxes(pdf_path))
        if proc.returncode != 0 and not pdf_ok:
            raise RuntimeError(
                f"Chrome print-to-pdf failed (rc={proc.returncode}): {proc.stderr[-800:]}"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    boxes = get_pdf_page_boxes(pdf_path)
    if not boxes:
        raise RuntimeError(f"Could not parse /MediaBox from generated PDF: {pdf_path}")
    for idx_b, (w, h) in enumerate(boxes, 1):
        if abs(w - 612.0) > 1.0 or abs(h - 792.0) > 1.0:
            raise RuntimeError(
                f"Page {idx_b} of {pdf_path} is not US Letter (612x792 pt): got {w}x{h} pt"
            )

    return {
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "lang": lang,
        "pages": len(boxes),
        "page_size_pt": boxes[0],
        "size_bytes": pdf_path.stat().st_size,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Render EN and ZH IEEE academic papers to US-Letter single-spaced PDFs via Headless Chrome."
    )
    ap.add_argument("--en-md", default="docs/EXPERIMENT_REPORT.md", help="English Markdown input")
    ap.add_argument("--zh-md", default="docs/EXPERIMENT_REPORT_ZH.md", help="Simplified Chinese Markdown input")
    ap.add_argument(
        "--en-pdf",
        default="docs/papers/Loop64_Technical_Report_EN.pdf",
        help="English US-Letter PDF output",
    )
    ap.add_argument(
        "--zh-pdf",
        default="docs/papers/Loop64_Technical_Report_ZH.pdf",
        help="Simplified Chinese US-Letter PDF output",
    )
    ap.add_argument("--chrome", default=None, help="Optional explicit path to Chrome binary")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results = []
    if args.en_md and args.en_pdf:
        results.append(
            render_pdf_from_markdown(
                Path(args.en_md),
                Path(args.en_pdf),
                lang="en",
                chrome_bin=args.chrome,
            )
        )
    if args.zh_md and args.zh_pdf:
        results.append(
            render_pdf_from_markdown(
                Path(args.zh_md),
                Path(args.zh_pdf),
                lang="zh",
                chrome_bin=args.chrome,
            )
        )
    import json

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
