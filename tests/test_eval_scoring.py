"""Answer extraction / scorer edge cases (pure python)."""
from __future__ import annotations

import math

import pytest

from qloop.eval import scoring as S

# ----------------------------------------------------------------------------- boxed / AIME

@pytest.mark.parametrize("text,expected", [
    (r"so the answer is \boxed{42}.", ["42"]),
    (r"\boxed{\frac{1}{2}} and later \boxed{3}", [r"\frac{1}{2}", "3"]),
    (r"\boxed{{a}+\{b\}}", [r"{a}+\{b\}"]),
    (r"\fbox{7}", ["7"]),
    (r"$\boxed 5$", ["5"]),
    ("no box here", []),
    (r"\boxed{unterminated", []),
])
def test_find_boxed(text, expected):
    assert S.find_boxed(text) == expected


@pytest.mark.parametrize("text,expected", [
    (r"... therefore \boxed{042}", 42),
    (r"\boxed{\text{the answer is } 117}", 117),
    (r"\boxed{x = 25}", 25),
    (r"\boxed{25.0}", 25),
    (r"\boxed{1000}", None),            # out of AIME range, no other number
    (r"\boxed{-3}", None),
    ("The final answer is 204.", 204),
    ("Final Answer: **73**", 73),
    ("Answer: $\\boxed{12}$", 12),
    ("blah 3 blah\n\nso we get 17 in total", 17),   # last-number fallback
    ("<think>the answer is 999 maybe</think>The answer is 5", 5),
    (r"\boxed{\frac{1}{2}} \boxed{7}", 7),
    ("", None),
    (r"\boxed{12^\circ}", 12),
])
def test_extract_aime(text, expected):
    assert S.extract_aime_answer(text) == expected


def test_aime_no_fallback_and_score():
    assert S.extract_aime_answer("we get 17", fallback_last_number=False) is None
    assert S.score_aime(r"\boxed{7}", "007") == 1.0
    assert S.score_aime(r"\boxed{8}", 7) == 0.0
    assert S.AIME.evaluate(r"\boxed{070}", {"answer": "70"}) == {"score": 1.0, "pred": 70}


@pytest.mark.parametrize("s,expected", [("1,000", None), ("12", 12), (" 0 ", 0), ("3/4", None), ("2.5", None),
                                        ("12%", 12), ("n = 9", 9), ("", None)])
def test_parse_int_answer(s, expected):
    assert S.parse_int_answer(s) == expected


# ----------------------------------------------------------------------------- math equivalence

@pytest.mark.parametrize("a,b", [
    ("0.5", r"\frac{1}{2}"), (r"\dfrac12", "1/2"), (r"\frac{3}{4}", "0.75"), ("1,000", "1000"),
    ("x = 5", "5"), (r"5\%", "5"), (r"90^\circ", "90"), ("2.50", "2.5"), (".5", "0.5"),
    (r"\left(1, 2\right)", "(1,2)"), (r"\{1,2,3\}", r"\{3,2,1\}"), ("1, 2", "2, 1"),
    (r"\sqrt{2}/2", r"\frac{\sqrt{2}}{2}"), ("x^2+2x+1", "(x+1)^2"), (r"2\pi", r"\pi \cdot 2"),
    (r"\text{5 cm}", "5"), ("10 dollars", "10"), (r"\frac{-1}{3}", "-1/3"), ("1e-3", "0.001"),
    (r"\boxed{4}", "4"), ("-0.25", r"-\frac14"),
])
def test_math_equal_true(a, b):
    assert S.math_equal(a, b), (a, b, S.normalize_math(a), S.normalize_math(b))


@pytest.mark.parametrize("a,b", [
    ("0.5", "0.6"), ("(1,2)", "(2,1)"), ("2x", "2"), ("x^2", "x^3"), (None, "1"), ("", "0"),
    ("1/0", "0"), ("(1,2)", "(1,2,3)"), (r"\sqrt{2}", "1.5"), ("yes", "no"),
])
def test_math_equal_false(a, b):
    assert not S.math_equal(a, b)


def test_extract_math_answer_and_scorer():
    assert S.extract_math_answer(r"<think>\boxed{9}</think> so \boxed{\dfrac{1}{3}}") == r"\frac{1}{3}"
    assert S.extract_math_answer("The final answer is 3/4 because reasons") == "3/4"
    assert S.extract_math_answer("nothing") is None
    assert S.score_math(r"\boxed{0.25}", r"\frac14") == 1.0
    assert S.MATH.score(r"\boxed{x+1}", {"answer": "1+x"}) == 1.0


# ----------------------------------------------------------------------------- MCQ

@pytest.mark.parametrize("text,n,expected", [
    ("Reasoning...\nAnswer: C", None, "C"),
    ("The correct answer is (B).", 4, "B"),
    ("**Answer:** D", None, "D"),
    (r"so \boxed{A}", None, "A"),
    (r"\boxed{\text{(E)}}", None, "E"),
    ("I pick option J", None, "J"),
    ("I pick option J", 4, None),          # J invalid for 4 choices
    ("Thinking about (A) and (B)...\n\nB", None, "B"),
    ("**C**", None, "C"),
    ("answer: e", None, None),            # lowercase letters are not answers
    ("", None, None),
    ("<think>Answer: A</think>Answer: D", None, "D"),
    ("Answer: A\nWait, actually Answer: B", None, "B"),
])
def test_extract_mcq(text, n, expected):
    assert S.extract_mcq(text, n) == expected


def test_mcq_scorer_uses_choices_count():
    item = {"choices": ["a", "b", "c", "d"], "answer": "B"}
    assert S.MCQ.score("Answer: B", item) == 1.0
    assert S.MCQ.score("Answer: H", item) == 0.0
    assert S.score_mcq("answer is (c)", "C") == 0.0 and S.score_mcq("answer is (C)", "c") == 1.0


# ----------------------------------------------------------------------------- ARC grids

def test_parse_grids_json_and_rows():
    t = "Input was\n1 2\n3 4\n\nmy answer: [[0, 1], [1, 0]]"
    assert S.parse_grids(t) == [[[1, 2], [3, 4]], [[0, 1], [1, 0]]]
    assert S.parse_grid(t) == [[0, 1], [1, 0]]
    rows = "Output:\n```\n012\n340\n```\n"
    assert S.parse_grid(rows) == [[0, 1, 2], [3, 4, 0]]
    multi = "[[1,2],\n [3,4]]"
    assert S.parse_grids(multi) == [[[1, 2], [3, 4]]]  # row-scan must not duplicate the JSON grid


def test_parse_grid_rejects_invalid():
    assert S.parse_grid("[[1, 2], [3]]") is None          # ragged
    assert S.parse_grid("[[10, 2], [3, 4]]") is None      # values > 9
    assert S.parse_grid("no grid") is None
    assert S.parse_grid("[[1,2],[3,4]") is None           # unterminated


def test_arc_attempts_and_pass2():
    gold = [[1, 1], [0, 0]]
    t = "Attempt 1: [[0,0],[1,1]]\nAttempt 2: [[1,1],[0,0]]"
    att = S.extract_arc_attempts(t, 2)
    assert att == [[[0, 0], [1, 1]], gold]
    assert S.arc_pass_at_2(att, gold) == 1.0
    assert S.arc_pass_at_2([[[0]], None, gold], gold) == 0.0  # third attempt does not count
    assert S.score_arc_task([[gold], [None, [[2]]]], [gold, [[3]]]) == 0.5
    assert S.ARC.evaluate("[[1,1],[0,0]]", {"answer": gold})["score"] == 1.0
    assert S.ARC.evaluate("[[1,1],[0,1]]", {"answer": gold})["score"] == 0.0


# ----------------------------------------------------------------------------- HLE / exact / GAIA

def test_hle_exact_and_mc():
    txt = "Explanation: because.\nExact Answer: The Eiffel Tower\nConfidence: 80%"
    item = {"answer": "Eiffel tower", "answer_type": "exactMatch"}
    ev = S.HLE.evaluate(txt, item)
    assert ev["score"] == 1.0 and ev["pred"] == "The Eiffel Tower" and ev["confidence"] == pytest.approx(0.8)
    assert S.HLE.score("Exact Answer: 0.5", {"answer": r"\frac{1}{2}", "answer_type": "exactMatch"}) == 1.0
    mc = {"answer": "C", "answer_type": "multipleChoice", "choices": None}
    assert S.HLE.score("Explanation: x\nAnswer: C. the third\nConfidence: 50", mc) == 1.0
    assert S.HLE.score("Explanation: x\nAnswer: B\nConfidence: 50", mc) == 0.0
    assert S.score_hle("**Answer:** C", mc) == 1.0
    assert S.extract_confidence("Confidence: 0.3") == pytest.approx(0.3)


def test_exact_match_normalisation():
    assert S.exact_match("The  Beatles!", "beatles")
    assert not S.exact_match("Rolling Stones", "beatles")
    assert not S.exact_match(None, "x")


@pytest.mark.parametrize("pred,gold,ok", [
    ("$1,000", "1000", True), ("17%", "17", True), ("Paris", "paris", True), ("St. Louis", "St Louis", True),
    ("a, b, 3", "a,b,3.0", True), ("a; b", "a, b", True), ("a, b", "a, b, c", False), ("3", "4", False),
    (None, "1", False),
])
def test_gaia_match(pred, gold, ok):
    assert S.gaia_match(pred, gold) is ok


def test_gaia_extract():
    assert S.extract_gaia_answer("thoughts...\nFINAL ANSWER: 42") == "42"
    assert S.GAIA.score("Final answer: Paris", {"answer": "paris"}) == 1.0
    assert S.extract_gaia_answer("no template") is None


def test_strip_thinking():
    assert S.strip_thinking("<think>a</think>b") == "b"
    assert S.strip_thinking("<think>unfinished") == "<think>unfinished"
    assert S.strip_thinking(None) == ""


# ----------------------------------------------------------------------------- pass@k / maj@k

def test_pass_at_k_matches_combinatorics():
    for n in range(1, 9):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                exact = 1.0 - math.comb(n - c, k) / math.comb(n, k)
                assert S.pass_at_k(n, c, k) == pytest.approx(exact)
    assert S.pass_at_k(10, 0, 5) == 0.0 and S.pass_at_k(10, 10, 1) == 1.0
    with pytest.raises(ValueError):
        S.pass_at_k(2, 1, 3)
    assert S.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)


def test_majority_vote_and_maj_at_k():
    assert S.majority_vote([1, 2, 2, None, 1, 2]) == 2
    assert S.majority_vote([3, 4]) == 3                          # tie -> first seen
    assert S.majority_vote([None, None]) is None
    assert S.majority_vote(["0.5", r"\frac12", "1"], equal=S.math_equal) == "0.5"
    assert S.maj_at_k([5, 7, 7], lambda a: a == 7) == 1.0
    assert S.maj_at_k([5, 7, 7], lambda a: a == 7, k=1) == 0.0


def test_get_scorer_registry():
    assert S.get_scorer("aime") is S.AIME
    assert S.get_scorer("lcb").name == "lcb"
    assert S.get_scorer("kernelbench").name == "kernelbench"
    with pytest.raises(KeyError, match="unknown scorer"):
        S.get_scorer("nope")
