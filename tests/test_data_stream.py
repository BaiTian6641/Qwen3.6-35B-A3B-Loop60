"""Packing, mixture weighting/resume, decontamination and the end-to-end train stream (offline)."""
import collections
import copy
import json

import pytest
import torch

from qloop.data import Decontaminator, MixtureStream, Packer, ToyTokenizer, build_train_stream
from qloop.data.formats import Example
from qloop.data.mixture import toy_cot_row
from qloop.data.packing import IGNORE, tokenize_example

# ----------------------------------------------------------------------------- packing


def _doc(n, start, trainable=True):
    ids = list(range(start, start + n))
    return ids, (list(ids) if trainable else [IGNORE] * n)


def test_packing_respects_boundaries_and_pads():
    p = Packer(seq_len=10, pad_id=0, respect_doc_boundaries=True)
    out = []
    out += p.add(*_doc(4, 100), doc_id=0)
    out += p.add(*_doc(5, 200), doc_id=1)   # 9 tokens buffered
    out += p.add(*_doc(3, 300), doc_id=2)   # does not fit -> row 0 padded, doc 2 starts row 1
    assert len(out) == 1
    r = out[0]
    assert r["input_ids"].shape == r["labels"].shape == r["doc_ids"].shape == r["position_ids"].shape == (10,)
    assert r["input_ids"].dtype == torch.long
    assert r["doc_ids"].tolist() == [0] * 4 + [1] * 5 + [-1]
    assert r["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0]
    assert r["labels"][-1].item() == IGNORE and r["input_ids"][-1].item() == 0
    assert r["n_pad"] == 1 and r["n_loss_tokens"] == 8 and r["n_docs"] == 2
    last = p.flush()
    assert last is not None and last["doc_ids"].tolist()[:3] == [2, 2, 2]


def test_packing_chunks_long_docs_and_continuous_mode():
    p = Packer(seq_len=8, pad_id=0, respect_doc_boundaries=True)
    out = p.add(*_doc(3, 10), doc_id=0)
    out += p.add(*_doc(20, 100), doc_id=1)  # longer than seq_len: fills current row, then chunks
    assert [r["doc_ids"].tolist() for r in out] == [[0, 0, 0, 1, 1, 1, 1, 1], [1] * 8]
    assert out[1]["position_ids"].tolist() == list(range(8))
    rest = p.flush()
    assert rest["doc_ids"].tolist() == [1] * 7 + [-1]
    all_ids = torch.cat([r["input_ids"] for r in [*out, rest]]).tolist()
    assert all_ids[:3 + 20] == list(range(10, 13)) + list(range(100, 120))

    c = Packer(seq_len=8, pad_id=0, respect_doc_boundaries=False)
    rows = []
    for k in range(5):
        rows += c.add(*_doc(5, 10 * k), doc_id=k)
    assert len(rows) == 3 and all(r["n_pad"] == 0 for r in rows)  # 25 tokens -> 3 full rows + 1 buffered
    assert rows[0]["doc_ids"].tolist() == [0] * 5 + [1] * 3


def test_packing_drops_rows_without_loss_and_state_roundtrip():
    p = Packer(seq_len=6, pad_id=0, respect_doc_boundaries=False)
    assert p.add(*_doc(6, 1, trainable=False), doc_id=0) == [] and p.n_dropped_no_loss == 1
    p.add(*_doc(4, 50), doc_id=1)
    sd = copy.deepcopy(p.state_dict())
    a = p.add(*_doc(4, 60), doc_id=2)
    q = Packer(seq_len=6, pad_id=0, respect_doc_boundaries=False)
    q.load_state_dict(json.loads(json.dumps(sd)))
    b = q.add(*_doc(4, 60), doc_id=2)
    assert len(a) == len(b) == 1 and torch.equal(a[0]["input_ids"], b[0]["input_ids"])


def test_tokenize_example_text_and_chat_separators():
    tok = ToyTokenizer()
    ids, labels = tokenize_example(Example("s", "text", text="abc"), tok)
    assert ids == [*tok.encode("abc"), tok.pad_token_id] and labels == ids  # <|endoftext|> separator, trained
    chat = Example("s", "chat", messages=[{"role": "user", "content": "q", "reasoning": None},
                                           {"role": "assistant", "content": "a", "reasoning": "r"}])
    ids, labels = tokenize_example(chat, tok)
    assert ids[-1] == tok.pad_token_id and labels[-1] == IGNORE
    assert tok.eos_token_id in [i for i, lab in zip(ids, labels) if lab != IGNORE]  # <|im_end|> trained


# ----------------------------------------------------------------------------- mixture

def _counter_factory(tag, n):
    def fac(epoch):
        return ((tag, epoch, i) for i in range(n))
    return fac


def test_mixture_weights_within_tolerance():
    m = MixtureStream([("a", 0.6, _counter_factory("a", 50)), ("b", 0.3, _counter_factory("b", 7)),
                       ("c", 0.1, _counter_factory("c", 1000))], seed=123)
    n = 20000
    counts = collections.Counter(next(m)[0] for _ in range(n))
    for name, w in [("a", 0.6), ("b", 0.3), ("c", 0.1)]:
        assert abs(counts[name] / n - w) < 0.015, (name, counts)
    st = m.stats()
    assert st["b"]["epoch"] >= 800  # small source re-opened many times (infinite stream)


def test_mixture_deterministic_and_epochs():
    def mk(seed):
        return MixtureStream([("a", 1.0, _counter_factory("a", 3)), ("b", 1.0, _counter_factory("b", 3))], seed=seed)
    x = [next(mk(7)) for _ in range(1)]
    s1, s2 = mk(7), mk(7)
    seq1, seq2 = [next(s1) for _ in range(50)], [next(s2) for _ in range(50)]
    assert seq1 == seq2 and x[0] == seq1[0]
    assert [next(mk(8)) for _ in range(50)] != seq1
    epochs_a = [item[1] for name, item in seq1 if name == "a"]
    assert epochs_a == sorted(epochs_a) and max(epochs_a) >= 1
    with pytest.raises(RuntimeError, match="produced no items"):
        next(MixtureStream([("empty", 1.0, lambda: iter(()))], seed=0))


def test_mixture_resume_identical():
    srcs = [("a", 0.5, _counter_factory("a", 13)), ("b", 0.3, _counter_factory("b", 5)),
            ("c", 0.2, _counter_factory("c", 29))]
    m = MixtureStream(srcs, seed=42)
    for _ in range(137):
        next(m)
    sd = json.loads(json.dumps(m.state_dict()))  # must be JSON-serialisable
    expected = [next(m) for _ in range(200)]
    r = MixtureStream(srcs, seed=42)
    r.load_state_dict(sd)
    assert [next(r) for _ in range(200)] == expected


def test_mixture_resume_with_cursor_factory():
    calls = []

    def fac(epoch, resume_from=None):
        calls.append(resume_from)
        start = 0 if resume_from is None else resume_from[0]
        for i in range(start, 40):
            for j in range(2):  # two items per raw row, like ARC test pairs
                ex = Example("s", "text", text=f"{epoch}-{i}-{j}", row_index=i, sub_index=j)
                if resume_from is not None and (i, j) <= tuple(resume_from):
                    continue
                yield ex

    m = MixtureStream([("s", 1.0, fac)], seed=0)
    for _ in range(31):
        next(m)
    sd = m.state_dict()
    assert sd["sources"]["s"]["cursor"] == [15, 0]
    exp = [next(m)[1].text for _ in range(100)]
    r = MixtureStream([("s", 1.0, fac)], seed=0)
    r.load_state_dict(sd)
    assert [next(r)[1].text for _ in range(100)] == exp
    assert calls[-2] == (15, 0)  # resumed via cursor, not by re-reading 31 items


# ----------------------------------------------------------------------------- decontam

AIME_LIKE = ("Patrick started walking at a constant rate along a straight road from school to the park. "
             "One hour after Patrick left, Tanya started running along the same road.")


def test_decontam_ngram_detection():
    d = Decontaminator(n=13)
    assert d.add_texts("aime_2026", [AIME_LIKE]) == 1
    leaked = "Solve: PATRICK started walking, at a constant rate along a straight road from school to the park!"
    assert d.is_contaminated(leaked) == (True, "aime_2026")
    assert d.is_contaminated("Patrick started walking at a constant rate along a road.") == (False, None)
    assert d.check(leaked, source="x")[0] and d.summary()["hits"] == {"aime_2026<-x": 1}


def test_decontam_short_texts_and_low_diversity():
    d = Decontaminator(n=13, short_min_words=6)
    d.add_texts("gpqa", ["Which particle mediates the weak nuclear force exactly?"])  # 8 words < 13
    assert d.is_contaminated("Q: which particle mediates the weak nuclear force exactly? A: W boson")[0]
    assert not d.is_contaminated("which particle mediates the strong force")[0]
    grid = "\n".join(["0000000000"] * 3 + ["0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1"])
    d.add_texts("arc_agi_2", [grid])
    assert not d.is_contaminated("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")[0]  # low-diversity grams ignored


def test_decontam_provider_failures():
    def boom():
        raise OSError("401 gated")
    with pytest.raises(RuntimeError, match="hle"):
        Decontaminator.from_providers({"hle": boom})
    d = Decontaminator.from_providers({"hle": boom, "aime": lambda: [AIME_LIKE]}, on_missing="warn")
    assert d.is_contaminated(AIME_LIKE)[0]


# ----------------------------------------------------------------------------- end to end

def _cfg(**over):
    cfg = {"data": {
        "seq_len": 1024, "respect_doc_boundaries": True, "streaming": True, "max_rows_per_source": None,
        "sources": [{"name": "toy_cot", "weight": 0.7, "toy_rows": 300, "bad_frac": 0.1},
                    {"name": "toy_text", "weight": 0.3, "toy_rows": 5}],
        "decontaminate": {"enabled": False},
        "log_every": 0,
    }}
    cfg["data"].update(over)
    return cfg


def test_build_train_stream_end_to_end():
    tok = ToyTokenizer()
    stream = build_train_stream(_cfg(), tok, seed=3)
    rows = [next(stream) for _ in range(40)]
    for r in rows:
        assert r["input_ids"].shape == (1024,) and r["labels"].shape == (1024,)
        assert int(r["input_ids"].max()) < tok.vocab_size and r["n_loss_tokens"] > 0
        lab = r["labels"]
        assert ((lab == IGNORE) | (lab == r["input_ids"])).all()
    text = tok.decode(torch.cat([r["input_ids"] for r in rows]), skip_special_tokens=False)
    assert "<think>\nStart with" in text or "<think>\nThe boxes hold" in text
    assert "gAAAA" not in text and "[redacted]" not in text  # bad_frac rows were filtered
    st = stream.stats()
    drops = st["drops"]["toy_cot"]
    assert drops["kept"] > 0 and sum(v for k, v in drops.items() if k.startswith("reasoning_")) > 0
    assert st["mixture"]["toy_text"]["epoch"] >= 1  # 5-row source wrapped around
    assert 0.5 < st["fill_ratio"] <= 1.0


def test_build_train_stream_resume_exact():
    tok = ToyTokenizer()
    cfg = _cfg(respect_doc_boundaries=False)
    a = build_train_stream(cfg, tok, seed=11)
    for _ in range(23):
        next(a)
    sd = json.loads(json.dumps(a.state_dict()))
    exp = [next(a) for _ in range(30)]
    b = build_train_stream(cfg, tok, seed=11)
    b.load_state_dict(sd)
    got = [next(b) for _ in range(30)]
    for x, y in zip(exp, got):
        assert torch.equal(x["input_ids"], y["input_ids"]) and torch.equal(x["labels"], y["labels"])
        assert torch.equal(x["doc_ids"], y["doc_ids"])


def test_build_train_stream_decontaminates():
    tok = ToyTokenizer()
    src_seed = 5 * 1009 + 0  # build_train_stream seeds source k with seed*1009 + k
    leaked = [toy_cot_row(src_seed, 0, i)["prompt"] for i in range(0, 300, 3)]
    cfg = _cfg(decontaminate={"enabled": True, "ngram": 8, "groups": ["toyeval"]},
               sources=[{"name": "toy_cot", "weight": 1.0, "toy_rows": 300}])
    stream = build_train_stream(cfg, tok, seed=5, eval_providers={"toyeval": lambda: leaked})
    rows = [next(stream) for _ in range(30)]
    text = tok.decode(torch.cat([r["input_ids"] for r in rows]), skip_special_tokens=False)
    assert stream.counters["decontaminated"] > 0
    assert not any(q in text for q in leaked)
    assert any(k.startswith("contaminated:toyeval") for k in stream.stats()["drops"]["toy_cot"])


def test_build_train_stream_config_errors():
    tok = ToyTokenizer()
    with pytest.raises(AssertionError, match="role 'rl'"):
        build_train_stream(_cfg(sources=[{"name": "kernelbench_train", "weight": 1.0}]), tok)
    with pytest.raises(AssertionError, match="not in the registry"):
        build_train_stream(_cfg(sources=[{"name": "mystery", "weight": 1.0}]), tok)
    with pytest.raises(AssertionError, match="unknown override"):
        build_train_stream(_cfg(sources=[{"name": "toy_cot", "weight": 1.0, "colour": "red"}]), tok)
    with pytest.raises(KeyError, match="decontam group 'nosuch'"):
        build_train_stream(_cfg(decontaminate={"enabled": True, "groups": ["nosuch"]}), tok)
