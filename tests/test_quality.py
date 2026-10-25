import numpy as np
import pytest

from src.quality import brier, ece, gains_curve, net_benefit, reliability_table


def test_brier_hand_case():
    assert brier([1, 0], [0.8, 0.2]) == pytest.approx(0.04)


def test_ece_hand_case():
    y = [1, 0, 1, 0]
    p = [0.9, 0.9, 0.1, 0.1]
    t = reliability_table(y, p, n_bins=2)
    assert list(t["n"]) == [2, 2]
    assert ece(y, p, n_bins=2) == pytest.approx(0.4)


def test_gains_perfect_ranking():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    score = [0.99, 0.98, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07]
    x, g = gains_curve(y, score)
    assert g[0] == 0.0
    assert x[2] == pytest.approx(0.2)
    assert g[2] == pytest.approx(1.0)
    assert g[-1] == pytest.approx(1.0)


def test_net_benefit_hand_case():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1]
    nb, nb_all = net_benefit(y, p, [0.25, 0.5])
    assert nb[0] == pytest.approx(0.5 - 0.25 * (0.25 / 0.75))
    assert nb[1] == pytest.approx(0.0)
    assert nb_all[0] == pytest.approx(0.5 - 0.5 * (0.25 / 0.75))