"""KernelBench subprocess evaluation (CPU) and LiveCodeBench-style code execution."""
from __future__ import annotations

import base64
import json
import pickle
import zlib

from qloop.eval.code_exec import decode_lcb_tests, extract_code, outputs_match, run_python_tests
from qloop.eval.kernelbench import evaluate_kernel, extract_kernel_source
from qloop.eval.scoring import get_scorer

REF = '''
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.lin = nn.Linear(16, 16)

    def forward(self, x):
        return torch.relu(self.lin(x)) * self.scale

def get_inputs():
    return [torch.randn(4, 16)]

def get_init_inputs():
    return [2.0]
'''

GOOD = '''
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.lin = nn.Linear(16, 16)   # same layout + same seed => same weights

    def forward(self, x):
        return torch.clamp(self.lin(x), min=0.0) * self.scale
'''

WRONG = GOOD.replace("torch.clamp(self.lin(x), min=0.0)", "self.lin(x)")
SHAPE = GOOD.replace("return torch.clamp(self.lin(x), min=0.0) * self.scale", "return x.sum(0)")


def test_kernelbench_correct_candidate():
    r = evaluate_kernel(REF, GOOD, device="cpu", n_correct_trials=3, n_perf_trials=3, warmup=1, timeout=120)
    assert r.compiled and r.correct, r.error or r.log
    assert r.trials_passed == 3 and r.speedup is not None and r.speedup > 0
    assert r.ref_ms is not None and r.new_ms is not None and r.extra["timer"] == "perf_counter"
    assert r.max_abs_diff is not None and r.max_abs_diff < 1e-5


def test_kernelbench_incorrect_and_broken_candidates():
    r = evaluate_kernel(REF, WRONG, device="cpu", n_correct_trials=3, timeout=120)
    assert r.compiled and not r.correct and "max abs diff" in r.error and r.speedup is None
    r = evaluate_kernel(REF, SHAPE, device="cpu", n_correct_trials=2, timeout=120)
    assert r.compiled and not r.correct and "shape mismatch" in r.error
    r = evaluate_kernel(REF, "class ModelNew(:\n", device="cpu", timeout=120)
    assert not r.compiled and r.stage == "compile" and "SyntaxError" in r.error
    r = evaluate_kernel(REF, "import torch\nclass Other: pass\n", device="cpu", timeout=120)
    assert not r.compiled and "ModelNew" in r.error
    r = evaluate_kernel(REF, "", device="cpu")
    assert not r.compiled and "empty" in r.error


def test_kernelbench_timeout():
    hang = "import time\nwhile True:\n    time.sleep(0.05)\n"
    r = evaluate_kernel(REF, hang, device="cpu", timeout=8)
    assert r.timed_out and not r.compiled and "timeout" in r.error


def test_kernelbench_scorer_extracts_code_block():
    text = f"Here is the optimized code:\n```python\n{GOOD}\n```\nDone."
    assert "class ModelNew" in extract_kernel_source(text)
    ev = get_scorer("kernelbench").evaluate(text, {"ref_src": REF, "device": "cpu", "n_perf_trials": 2,
                                                   "timeout": 120})
    assert ev["score"] == 1.0 and ev["compiled"] and ev["correct"]
    assert get_scorer("kernelbench").evaluate("no code", {"ref_src": REF})["error"] == "no code block found"


# ----------------------------------------------------------------------------- code exec (LCB)

def test_extract_code_variants():
    assert extract_code("```python\nprint(1)\n```\ntext\n```python\nprint(2)\n```") == "print(2)"
    assert extract_code("```\nx=1\n```") == "x=1"
    assert extract_code("```python\nprint('unterminated')") == "print('unterminated')"
    assert extract_code("```cpp\nint main(){}\n```") == "int main(){}"  # fallback to any block
    assert extract_code("no code") is None
    assert extract_code("<think>```python\nbad\n```</think>```python\ngood\n```") == "good"


def test_outputs_match():
    assert outputs_match("1 2\n3\n", "1 2\n3")
    assert outputs_match("0.3333333333", "0.333333333")
    assert not outputs_match("1 2", "1 3")


def test_run_python_tests_stdin_and_functional():
    stdin_tests = [{"input": "3\n1 2 3\n", "output": "6\n", "testtype": "stdin"},
                   {"input": "2\n5 5\n", "output": "10\n", "testtype": "stdin"}]
    ok = "n = int(input())\nprint(sum(map(int, input().split())))\n"
    assert run_python_tests(ok, stdin_tests)["all_passed"]
    bad = run_python_tests("print(0)", stdin_tests)
    assert not bad["all_passed"] and "wrong answer" in bad["first_error"]
    crash = run_python_tests("raise SystemExit(3)", stdin_tests)
    assert "exit code 3" in crash["first_error"]
    hang = run_python_tests("while True: pass", stdin_tests[:1], timeout=1.0)
    assert "timeout" in hang["first_error"]

    func_tests = [{"input": "[1, 2, 3]\n2", "output": "[3, 4, 5]", "testtype": "functional"}]
    sol = ("class Solution:\n    def addAll(self, nums: List[int], k: int) -> List[int]:\n"
           "        return [x + k for x in nums]\n")
    assert run_python_tests(sol, func_tests, fn_name="addAll")["all_passed"]
    assert not run_python_tests(sol.replace("x + k", "x"), func_tests, fn_name="addAll")["all_passed"]
    assert "func_name" in run_python_tests(sol, func_tests)["first_error"]


def test_lcb_private_decoding_and_scorer():
    tests = [{"input": "", "output": "hi\n", "testtype": "stdin"}]
    enc = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert decode_lcb_tests(enc) == tests
    assert decode_lcb_tests(json.dumps(tests)) == tests and decode_lcb_tests("") == []
    item = {"public_tests": [], "private_tests": tests, "fn_name": None}
    sc = get_scorer("lcb")
    assert sc.evaluate("```python\nprint('hi')\n```", item)["score"] == 1.0
    assert sc.evaluate("```python\nprint('ho')\n```", item)["score"] == 0.0
    assert sc.evaluate("nothing", item)["error"] == "no code block found"
