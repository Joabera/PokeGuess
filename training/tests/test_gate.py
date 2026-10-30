import numpy as np
import pytest

from gate import BARS, confidence_summary, decide


def test_bars_match_spec():
    assert BARS == {"clean_all": 95.0, "sketch_all": 90.0, "clean_gen13": 95.7, "sketch_gen13": 92.8}


def test_decide_pass_at_bars():
    passed, lines = decide(dict(BARS))
    assert passed
    assert any("PASS" in line for line in lines)


@pytest.mark.parametrize("metric", list(BARS))
def test_decide_fails_each_bar(metric):
    metrics = dict(BARS)
    metrics[metric] -= 0.1
    passed, lines = decide(metrics)
    assert not passed
    assert any(metric in line and "FAIL" in line for line in lines)


def test_confidence_summary():
    conf = [0.9, 0.85, 0.5, 0.99, 0.79]
    s = confidence_summary(conf)
    assert s["share_ge_80"] == 60.0
    assert s["p10"] == pytest.approx(float(np.percentile(conf, 10)))
