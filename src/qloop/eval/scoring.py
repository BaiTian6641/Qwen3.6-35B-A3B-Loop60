"""Answer extraction + scorers (pure python; sympy used only if importable).

Every scorer follows one interface (`Scorer`):
    extract(text, item) -> canonical answer | None      (hashable => usable for maj@k)
    match(pred, item)   -> bool
    evaluate(text, item) -> {"score": float, "pred": ..., ...extra fields}

Text produced by thinking models ("<think> ... </think> answer") is reduced to the part after
the last `</think>` before extraction (if the think block never closed, the whole text is used).
"""
from __future__ import annotations

import difflib
import json
import logging
import math
import random
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable

log = logging.getLogger("qloop")

Grid = list  # list[list[int]]


# ============================================================================ generic helpers

def strip_thinking(text: str | None) -> str:
    if not text:
        return ""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):] if idx >= 0 else text


def _match_brace(s: str, open_idx: int) -> int:
    """Index of the brace closing s[open_idx] == '{', or -1."""
    depth = 0
    for i in range(open_idx, len(s)):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in "{}":
            continue
        if c == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif c == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return i
    return -1


def find_boxed(text: str) -> list[str]:
    """Contents of every \\boxed{...} / \\fbox{...} (nested braces ok) and `\\boxed 5`, in order."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*", text or ""):
        i = m.end()
        if i < len(text) and text[i] == "{":
            j = _match_brace(text, i)
            if j > i:
                out.append(text[i + 1:j])
        elif m.group(0).endswith((" ", "\t")) or (i < len(text) and text[i] not in "{"):
            tok = re.match(r"[^\s$]+", text[i:])
            if tok:
                out.append(tok.group(0))
    return out


def last_boxed(text: str) -> str | None:
    b = find_boxed(text)
    return b[-1] if b else None


def _unwrap_cmd(s: str, cmd: str) -> str:
    """\\text{abc} -> abc (brace-aware, repeated)."""
    pat = "\\" + cmd + "{"
    while True:
        i = s.find(pat)
        if i < 0:
            return s
        j = _match_brace(s, i + len(pat) - 1)
        if j < 0:
            return s
        s = s[:i] + s[i + len(pat):j] + s[j + 1:]


def clean_answer(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    for a, b in (("\\(", ""), ("\\)", ""), ("\\[", ""), ("\\]", ""), ("$", ""), ("\\displaystyle", "")):
        s = s.replace(a, b)
    for cmd in ("textbf", "mathbf", "mathrm", "text", "textrm", "mbox", "boxed"):
        s = _unwrap_cmd(s, cmd)
    s = s.replace("**", "").strip()
    s = s.rstrip(".").strip()
    return s


# ============================================================================ AIME (integer 0..999)

_INT_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?![\d,])|[+-]?\d+")


def parse_int_answer(s: Any, lo: int = 0, hi: int = 999) -> int | None:
    s = clean_answer(s)
    if not s:
        return None
    s = re.sub(r"\^\s*\{?\\circ\}?|°|\\%|%", "", s).strip()
    if "=" in s:
        s = s.rsplit("=", 1)[1].strip()
    m = re.fullmatch(r"([+-]?\d{1,3}(?:,\d{3})+|[+-]?\d+)(?:\.0*)?", s.replace(" ", ""))
    if m:
        v = int(m.group(1).replace(",", ""))
    else:
        nums = _INT_RE.findall(s)
        if len(nums) != 1 or re.search(r"\d\.\d*[1-9]", s) or "/" in s or "frac" in s:
            return None
        v = int(nums[0].replace(",", ""))
    return v if lo <= v <= hi else None


_FINAL_PATTERNS = [
    re.compile(r"(?i)final\s+answer\s*(?:is|:|=)?\s*[:：]?\s*(.+)"),
    re.compile(r"(?i)(?:the\s+)?answer\s*(?:is|:|=)\s*[:：]?\s*(.+)"),
]


def extract_final_answer_phrase(text: str) -> str | None:
    """Text following the last 'final answer is' / 'answer:' phrase (one line)."""
    t = strip_thinking(text)
    for pat in _FINAL_PATTERNS:
        ms = list(pat.finditer(t))
        if ms:
            return ms[-1].group(1).strip().split("\n")[0].strip()
    return None


def extract_aime_answer(text: str, fallback_last_number: bool = True) -> int | None:
    t = strip_thinking(text)
    for b in reversed(find_boxed(t)):
        v = parse_int_answer(b)
        if v is not None:
            return v
    phrase = extract_final_answer_phrase(t)
    if phrase is not None:
        m = re.match(r"[\s*$\\(]*([+-]?\d{1,3}(?:,\d{3})+|[+-]?\d+)(?![\d./])", phrase)
        if m:
            v = parse_int_answer(m.group(1))
            if v is not None:
                return v
    if fallback_last_number:
        tail = t.strip().split("\n\n")[-1][-400:]
        nums = [n for n in _INT_RE.findall(tail)]
        for n in reversed(nums):
            v = parse_int_answer(n)
            if v is not None:
                return v
    return None


def score_aime(text: str, gold: Any) -> float:
    g = parse_int_answer(gold)
    p = extract_aime_answer(text)
    return float(g is not None and p == g)


# ============================================================================ math equivalence

def _split_top(s: str) -> tuple[str, list[str]]:
    """'(1,2)' -> ('()', ['1','2']); 'a,b' -> ('', ['a','b']). Top-level commas only."""
    br = ""
    if len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        inner, depth, ok = s[1:-1], 0, True
        for c in inner:
            depth += c in "([{"
            depth -= c in ")]}"
            if depth < 0:
                ok = False
                break
        if ok and depth == 0:
            br, s = s[0] + s[-1], inner
    parts, depth, cur = [], 0, []
    for c in s:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return br, [p for p in parts if p != ""]


def normalize_math(s: Any) -> str:
    s = clean_answer(s)
    # drop trailing unit words ("5 cm", "12 dollars"); before \pi -> pi so "3 \pi" survives
    m = re.fullmatch(r"\s*([-+]?[\d.,/]+)\s+[A-Za-z]{2,}[A-Za-z\s.]*", s)
    if m:
        s = m.group(1)
    for a, b in (("\\{", "{"), ("\\}", "}"),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("\\ ", ""), ("\\left", ""),
                 ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\cdot", "*"),
                 ("\\times", "*"), ("\\leq", "<="), ("\\geq", ">="), ("\\le", "<="), ("\\ge", ">="),
                 ("°", ""), ("\\infty", "oo"),
                 ("\\pi", "pi"), ("\n", " ")):
        s = s.replace(a, b)
    s = re.sub(r"\^\s*\{?\\circ\}?", "", s)
    s = re.sub(r"\\%|%", "", s)
    # "x = 5" / "a=5" -> "5" (single assignment only)
    if s.count("=") == 1 and re.match(r"^\s*[A-Za-z](?:_\{?\w+\}?)?\s*=", s):
        s = s.split("=", 1)[1]
    s = s.replace(" ", "")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    if s.startswith("."):
        s = "0" + s
    s = re.sub(r"(?<![\d])\.(\d)", r"0.\1", s)
    while len(s) >= 2 and s[0] == "{" and _match_brace(s, 0) == len(s) - 1:
        s = s[1:-1]
    s = s.rstrip(".")
    if re.fullmatch(r"[-+]?\d+\.\d*?0+", s):  # 2.50 -> 2.5, 3.0 -> 3
        s = s.rstrip("0").rstrip(".")
    return s


def parse_number(s: Any) -> Fraction | None:
    s = normalize_math(s)
    if not s:
        return None
    sign = -1 if s.startswith("-") else 1
    body = s[1:] if s[:1] in "+-" else s
    m = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", body)
    if m:
        a, b = parse_number(m.group(1)), parse_number(m.group(2))
        if a is None or b is None or b == 0:
            return None
        return sign * a / b
    m = re.fullmatch(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", body)
    if m:
        den = Fraction(m.group(2))
        return None if den == 0 else sign * Fraction(m.group(1)) / den
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+", body):
        try:
            return sign * Fraction(body)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _latex_to_sympy_str(s: str) -> str:
    s = s.replace("\\frac", "\\FRAC")
    for _ in range(20):
        i = s.find("\\FRAC{")
        if i < 0:
            break
        j = _match_brace(s, i + 5)
        if j < 0 or j + 1 >= len(s) or s[j + 1] != "{":
            break
        k = _match_brace(s, j + 1)
        if k < 0:
            break
        s = s[:i] + "((" + s[i + 6:j] + ")/(" + s[j + 2:k] + "))" + s[k + 1:]
    for _ in range(20):
        m = re.search(r"\\sqrt\[([^\]]+)\]\{", s)
        if not m:
            break
        j = _match_brace(s, m.end() - 1)
        if j < 0:
            break
        s = s[:m.start()] + "((" + s[m.end():j] + ")**(1/(" + m.group(1) + ")))" + s[j + 1:]
    for _ in range(20):
        i = s.find("\\sqrt{")
        if i < 0:
            break
        j = _match_brace(s, i + 5)
        if j < 0:
            break
        s = s[:i] + "sqrt(" + s[i + 6:j] + ")" + s[j + 1:]
    s = s.replace("^", "**").replace("{", "(").replace("}", ")").replace("\\", "")
    return s


def _sympy_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    if len(a) > 200 or len(b) > 200:
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:
        return False
    tr = (*standard_transformations, implicit_multiplication_application)
    try:
        ea = parse_expr(_latex_to_sympy_str(a), transformations=tr, evaluate=True)
        eb = parse_expr(_latex_to_sympy_str(b), transformations=tr, evaluate=True)
        diff = ea - eb
        if diff == 0:
            return True
        syms = sorted(ea.free_symbols | eb.free_symbols, key=str)
        if not syms:
            v = complex(sympy.N(diff))
            return abs(v) <= tol * max(1.0, abs(complex(sympy.N(eb))))
        rng = random.Random(0)
        for _ in range(4):
            sub = {x: rng.uniform(0.5, 2.0) for x in syms}
            va, vb = complex(ea.evalf(subs=sub)), complex(eb.evalf(subs=sub))
            if not abs(va - vb) <= tol * max(1.0, abs(vb)):
                return False
        return True
    except Exception:
        return False


def math_equal(pred: Any, gold: Any, rel_tol: float = 1e-6, _depth: int = 0) -> bool:
    if pred is None or gold is None:
        return False
    a, b = normalize_math(pred), normalize_math(gold)
    if not a:
        return False
    if a == b or a.lower() == b.lower():
        return True
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None:
        return na == nb or abs(float(na) - float(nb)) <= rel_tol * max(1.0, abs(float(nb)))
    if _depth < 3:
        bra, pa = _split_top(a)
        brb, pb = _split_top(b)
        if (len(pa) > 1 or len(pb) > 1) and len(pa) == len(pb):
            if bra == brb and bra in ("()", "[]"):  # ordered tuples / intervals
                return all(math_equal(x, y, rel_tol, _depth + 1) for x, y in zip(pa, pb))
            if bra == brb or not bra or not brb:  # sets / plain lists: order-insensitive
                left = list(pb)
                for x in pa:
                    hit = next((k for k, y in enumerate(left) if math_equal(x, y, rel_tol, _depth + 1)), None)
                    if hit is None:
                        return False
                    left.pop(hit)
                return True
            return False
    if na is not None or nb is not None or re.search(r"[A-Za-z\\]", a + b):
        return _sympy_equal(a, b, rel_tol)
    return False


def extract_math_answer(text: str) -> str | None:
    t = strip_thinking(text)
    b = last_boxed(t)
    if b is not None:
        return normalize_math(b)
    phrase = extract_final_answer_phrase(t)
    if phrase:
        phrase = re.split(r"\s(?:because|since|as)\s", phrase)[0]
        return normalize_math(phrase)
    return None


def score_math(text: str, gold: Any) -> float:
    return float(math_equal(extract_math_answer(text), gold))


# ============================================================================ multiple choice

LETTERS = "ABCDEFGHIJ"


def extract_mcq(text: str, n_choices: int | None = None, letters: str = LETTERS,
                allow_leading: bool = False) -> str | None:
    """Last answer letter. `allow_leading` also accepts a text that merely *starts* with a letter
    ("C. because ...") - only safe on an already isolated answer line."""
    valid = letters[:n_choices] if n_choices else letters
    t = strip_thinking(text)
    if not t.strip():
        return None
    L = f"[{valid}]"
    pats = [
        rf"\\boxed\{{\s*(?:\\(?:text|textbf|mathrm|mathbf)\{{)?\s*\(?({L})\)?",
        rf"(?i:answer)[*_\s]*(?:(?i:is)\b|[:：=])[*_\s]*(?:(?i:option|choice)\b)?[*_\s]*[\(\[]?\s*({L})\s*[\)\]]?(?![A-Za-z])",
        rf"(?i:answer)\s+[\(\[]({L})[\)\]]",
        rf"(?i:option|choice)\s+[\(\[]?({L})[\)\]]?(?![A-Za-z])",
        rf"\*\*\s*[\(\[]?({L})[\)\]]?[.:]?\s*\*\*",
        rf"(?m)^\s*[\(\[]?({L})[\)\]]?[.:]?\s*$",
        rf"[\(\[]({L})[\)\]]",
    ]
    if allow_leading:
        pats.append(rf"^\s*[\(\[]?({L})(?:[\)\].:,]|\s|$)")
    for p in pats:
        ms = list(re.finditer(p, t))
        if ms:
            return ms[-1].group(1).upper()
    return None


def score_mcq(text: str, gold: str, n_choices: int | None = None) -> float:
    p = extract_mcq(text, n_choices)
    return float(p is not None and p == str(gold).strip().upper()[:1])


# ============================================================================ ARC-AGI grids

def _valid_grid(g: Any) -> bool:
    return (isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) > 0 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


def _json_grids(text: str) -> list[tuple[int, int, Grid]]:
    out = []
    i = 0
    while True:
        i = text.find("[", i)
        if i < 0:
            break
        m = re.match(r"\[\s*\[", text[i:])
        if not m:
            i += 1
            continue
        depth, j = 0, i
        while j < len(text):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            break
        try:
            g = json.loads(text[i:j + 1])
        except ValueError:
            g = None
        if _valid_grid(g):
            out.append((i, j + 1, g))
            i = j + 1
        else:
            i += 1
    return out


_ROW_RE = re.compile(r"^[\s|,\[\]]*\d(?:[\s|,]*\d)*[\s|,\[\]]*$")


def _row_grids(text: str) -> list[tuple[int, Grid]]:
    out, cur, start, pos = [], [], 0, 0
    for line in text.split("\n"):
        s = line.strip().strip("`")
        is_row = bool(s) and bool(_ROW_RE.match(s))
        if is_row:
            toks = re.findall(r"\d+", s)
            row = [int(c) for c in s if c.isdigit()] if len(toks) == 1 else [int(x) for x in toks]
            if any(v > 9 for v in row):
                is_row = False
        if is_row:
            if cur and len(row) != len(cur[0]):
                out.append((start, cur))
                cur = []
            if not cur:
                start = pos
            cur.append(row)
        elif cur:
            out.append((start, cur))
            cur = []
        pos += len(line) + 1
    if cur:
        out.append((start, cur))
    return [(p, g) for p, g in out if _valid_grid(g) and (len(g) > 1 or len(g[0]) > 1)]


def parse_grids(text: str) -> list[Grid]:
    """All grids in the text (JSON list-of-lists, or blocks of digit rows), in order of appearance."""
    t = strip_thinking(text)
    js = _json_grids(t)
    found = [(p, g) for p, _, g in js]
    for p, g in _row_grids(t):
        if not any(a <= p < b for a, b, _g in js):
            found.append((p, g))
    return [g for _, g in sorted(found, key=lambda x: x[0])]


def parse_grid(text: str) -> Grid | None:
    gs = parse_grids(text)
    return gs[-1] if gs else None


def extract_arc_attempts(text: str, k: int = 2) -> list[Grid]:
    """Last k distinct grids (for prompts that ask for several attempts in one answer)."""
    out: list[Grid] = []
    for g in reversed(parse_grids(text)):
        if g not in out:
            out.append(g)
        if len(out) == k:
            break
    return list(reversed(out))


def grid_equal(a: Any, b: Any) -> bool:
    return _valid_grid(a) and _valid_grid(b) and a == b


def arc_pass_at_2(attempts: Sequence[Grid | None], gold: Grid) -> float:
    return float(any(grid_equal(a, gold) for a in list(attempts)[:2]))


def score_arc_task(attempts_per_test: Sequence[Sequence[Grid | None]], golds: Sequence[Grid]) -> float:
    """Official ARC-AGI task score: fraction of test outputs solved within 2 attempts each."""
    if not golds:
        return 0.0
    return sum(arc_pass_at_2(a, g) for a, g in zip(attempts_per_test, golds)) / len(golds)


# ============================================================================ HLE / exact match

_DROP_PUNCT = set(string.punctuation) - {".", "-", "/"}


def normalize_text(s: Any) -> str:
    s = clean_answer(s).lower()
    s = "".join(c for c in s if c not in _DROP_PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).rstrip(".")


def extract_labelled(text: str, label: str = r"(?:exact\s+)?answer") -> str | None:
    t = strip_thinking(text)
    ms = list(re.finditer(rf"(?im)^[\s*_#>-]*{label}[\s*_]*[:：][\s*_]*(.+?)\s*$", t))
    if ms:
        return ms[-1].group(1).strip().strip("*").strip()
    return None


def extract_hle_answer(text: str) -> str | None:
    a = extract_labelled(text)
    if a:
        return a
    b = last_boxed(strip_thinking(text))
    if b is not None:
        return b.strip()
    lines = [ln.strip() for ln in strip_thinking(text).strip().split("\n") if ln.strip()]
    return lines[-1] if lines else None


def extract_confidence(text: str) -> float | None:
    a = extract_labelled(text, "confidence")
    if not a:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", a)
    if not m:
        return None
    v = float(m.group(1))
    return v / 100.0 if v > 1 or "%" in a else v


def exact_match(pred: Any, gold: Any) -> bool:
    if pred is None:
        return False
    if normalize_text(pred) == normalize_text(gold) and normalize_text(gold):
        return True
    return math_equal(pred, gold)


def score_hle(text: str, item: dict[str, Any]) -> float:
    return HLE.evaluate(text, item)["score"]


# ============================================================================ GAIA (official quasi exact match)

def _gaia_num(s: str) -> float | None:
    s = str(s)
    for ch in ("$", "%", ","):
        s = s.replace(ch, "")
    try:
        return float(s.strip())
    except ValueError:
        return None


def _gaia_str(s: str, remove_punct: bool = True) -> str:
    s = re.sub(r"\s", "", str(s))
    if remove_punct:
        s = s.translate(str.maketrans("", "", string.punctuation))
    return s.lower()


def gaia_match(pred: Any, gold: Any) -> bool:
    if pred is None:
        return False
    pred, gold = str(pred), str(gold)
    if _gaia_num(gold) is not None:
        return _gaia_num(pred) == _gaia_num(gold)
    if any(c in gold for c in ",;"):
        gl = [x.strip() for x in re.split(r"[,;]", gold)]
        pl = [x.strip() for x in re.split(r"[,;]", pred)]
        if len(gl) != len(pl):
            return False
        for p, g in zip(pl, gl):
            if _gaia_num(g) is not None:
                if _gaia_num(p) != _gaia_num(g):
                    return False
            elif _gaia_str(p, False) != _gaia_str(g, False):
                return False
        return True
    return _gaia_str(pred) == _gaia_str(gold)


def extract_gaia_answer(text: str) -> str | None:
    t = strip_thinking(text)
    ms = list(re.finditer(r"(?i)final\s+answer\s*[:：]\s*(.+)", t))
    if ms:
        return ms[-1].group(1).strip().strip("*").strip()
    return None


# ============================================================================ MRCR (OpenAI 1M retrieval)

def extract_mrcr_answer(text: str, random_string_to_prepend: str = "") -> str | None:
    """Strip `<think>...</think>` reasoning; trim leading whitespace when locating the required prefix."""
    pred = strip_thinking(text)
    prefix = str(random_string_to_prepend or "")
    if prefix:
        if not pred.startswith(prefix):
            pred = pred.lstrip()
    else:
        pred = pred.strip()
    return pred if pred != "" else None


def score_mrcr(text: str, gold: Any, random_string_to_prepend: str = "") -> float:
    """OpenAI official MRCR grading: 0.0 if `random_string_to_prepend` is missing after `<think>` stripping,
    else `difflib.SequenceMatcher(None, pred[len(prefix):], gold[len(prefix):]).ratio()`."""
    pred = extract_mrcr_answer(text, random_string_to_prepend)
    if pred is None:
        return 0.0
    prefix = str(random_string_to_prepend or "")
    gold_s = "" if gold is None else str(gold)
    if prefix and not pred.startswith(prefix):
        return 0.0
    pred_body = pred[len(prefix):]
    gold_body = gold_s[len(prefix):] if (prefix and gold_s.startswith(prefix)) else gold_s
    return float(difflib.SequenceMatcher(None, pred_body, gold_body).ratio())


# ============================================================================ pass@k / maj@k

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c, k) / C(n, k), numerically stable."""
    if k <= 0 or n <= 0:
        raise ValueError(f"need n > 0 and k > 0, got n={n} k={k}")
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def pass_hat_k(n: int, c: int, k: int) -> float:
    """tau-bench pass^k: probability that k i.i.d. trials ALL succeed = C(c,k)/C(n,k)."""
    if k > n or k <= 0:
        raise ValueError(f"need 0 < k <= n, got n={n} k={k}")
    return math.comb(c, k) / math.comb(n, k)


def majority_vote(answers: Sequence[Any], equal: Callable[[Any, Any], bool] | None = None) -> Any:
    """Most common non-None answer (ties -> first seen). `equal` clusters equivalent answers."""
    clusters: list[list[Any]] = []
    for a in answers:
        if a is None:
            continue
        for cl in clusters:
            if (equal(a, cl[0]) if equal else a == cl[0]):
                cl.append(a)
                break
        else:
            clusters.append([a])
    if not clusters:
        return None
    best = max(range(len(clusters)), key=lambda i: (len(clusters[i]), -i))
    return clusters[best][0]


def maj_at_k(answers: Sequence[Any], is_correct: Callable[[Any], bool], k: int | None = None,
             equal: Callable[[Any, Any], bool] | None = None) -> float:
    ans = list(answers)[: k or len(answers)]
    winner = majority_vote(ans, equal)
    return float(winner is not None and bool(is_correct(winner)))


# ============================================================================ Scorer objects

@dataclass(frozen=True)
class Scorer:
    name: str
    extract: Callable[[str, dict[str, Any]], Any]
    match: Callable[[Any, dict[str, Any]], bool]
    equal: Callable[[Any, Any], bool] | None = None          # answer clustering for maj@k
    evaluate_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None

    def evaluate(self, text: str, item: dict[str, Any]) -> dict[str, Any]:
        if self.evaluate_fn is not None:
            return self.evaluate_fn(text, item)
        pred = self.extract(text, item)
        ok = pred is not None and bool(self.match(pred, item))
        return {"score": float(ok), "pred": pred}

    def score(self, text: str, item: dict[str, Any]) -> float:
        return float(self.evaluate(text, item)["score"])


def _n_choices(item: dict[str, Any]) -> int | None:
    ch = item.get("choices")
    return len(ch) if isinstance(ch, (list, tuple)) and ch else item.get("n_choices")


def _hle_extract(text: str, item: dict[str, Any]) -> Any:
    raw = extract_hle_answer(text)
    if str(item.get("answer_type", "")).lower().startswith("multiple"):
        return extract_mcq(raw, _n_choices(item), allow_leading=True) if raw else None
    return raw


def _hle_match(pred: Any, item: dict[str, Any]) -> bool:
    if str(item.get("answer_type", "")).lower().startswith("multiple"):
        return str(pred).upper() == str(item["answer"]).strip().upper()[:1]
    return exact_match(pred, item["answer"])


def _hle_eval(text: str, item: dict[str, Any]) -> dict[str, Any]:
    pred = _hle_extract(text, item)
    ok = pred is not None and _hle_match(pred, item)
    return {"score": float(ok), "pred": pred, "confidence": extract_confidence(text)}


def _arc_eval(text: str, item: dict[str, Any]) -> dict[str, Any]:
    pred = parse_grid(text)
    gold = item.get("answer")
    return {"score": float(gold is not None and grid_equal(pred, gold)),
            "pred": json.dumps(pred) if pred is not None else None}


def _mrcr_extract(text: str, item: dict[str, Any]) -> str | None:
    return extract_mrcr_answer(text, str(item.get("random_string_to_prepend") or ""))


def _mrcr_eval(text: str, item: dict[str, Any]) -> dict[str, Any]:
    prefix = str(item.get("random_string_to_prepend") or "")
    pred = extract_mrcr_answer(text, prefix)
    gold = "" if item.get("answer") is None else str(item["answer"])
    if pred is None or (prefix and not pred.startswith(prefix)):
        return {"score": 0.0, "pred": pred, "prefix_ok": False}
    pred_body = pred[len(prefix):]
    gold_body = gold[len(prefix):] if (prefix and gold.startswith(prefix)) else gold
    ratio = float(difflib.SequenceMatcher(None, pred_body, gold_body).ratio())
    return {"score": ratio, "pred": pred, "prefix_ok": True}


AIME = Scorer("aime", lambda t, it: extract_aime_answer(t), lambda p, it: p == parse_int_answer(it["answer"]))
MATH = Scorer("math", lambda t, it: extract_math_answer(t), lambda p, it: math_equal(p, it["answer"]),
              equal=math_equal)
MCQ = Scorer("mcq", lambda t, it: extract_mcq(t, _n_choices(it)),
             lambda p, it: p == str(it["answer"]).strip().upper()[:1])
HLE = Scorer("hle", _hle_extract, _hle_match, equal=exact_match, evaluate_fn=_hle_eval)
ARC = Scorer("arc", lambda t, it: (json.dumps(parse_grid(t)) if parse_grid(t) is not None else None),
             lambda p, it: grid_equal(json.loads(p), it.get("answer")), evaluate_fn=_arc_eval)
GAIA = Scorer("gaia", lambda t, it: extract_gaia_answer(t), lambda p, it: gaia_match(p, it["answer"]),
              equal=gaia_match)
EXACT = Scorer("exact", lambda t, it: extract_hle_answer(t), lambda p, it: exact_match(p, it["answer"]),
               equal=exact_match)
MRCR = Scorer("mrcr", _mrcr_extract,
              lambda p, it: score_mrcr(p or "", it.get("answer"), str(it.get("random_string_to_prepend") or "")) >= 1.0,
              evaluate_fn=_mrcr_eval)

SCORERS: dict[str, Scorer] = {s.name: s for s in (AIME, MATH, MCQ, HLE, ARC, GAIA, EXACT, MRCR)}


def get_scorer(name: str) -> Scorer:
    if name in SCORERS:
        return SCORERS[name]
    if name == "lcb":
        from .code_exec import LCB_SCORER
        SCORERS[name] = LCB_SCORER
        return LCB_SCORER
    if name == "kernelbench":
        from .kernelbench import KERNELBENCH_SCORER
        SCORERS[name] = KERNELBENCH_SCORER
        return KERNELBENCH_SCORER
    raise KeyError(f"unknown scorer {name!r}; available: {sorted([*SCORERS, 'lcb', 'kernelbench'])}")
