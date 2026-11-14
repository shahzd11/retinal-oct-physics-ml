"""Fails if any transformer learned from data (POD modes, residual POD, PCA, scalers inside the
classifier) is fitted on rows of the test fold."""
import numpy as np
import pytest

from retina.ensemble import lr
from retina.hybrid import HybridClassifier, assemble
from retina.image_rom import ImagePOD
from retina.v6eval import assert_no_leak, folds, oof_proba, pod_sweep_oof, residual_oof


def data(n=90, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["NORMAL", "AMD", "DME"])[np.arange(n) % 3]
    groups = np.arange(n) // 2                       # 2 scans per patient
    T = rng.normal(size=(n, 64)) + (y == "AMD")[:, None] * 0.5
    T[rng.random(T.shape) < 0.05] = np.nan           # masked pixels
    F = rng.normal(size=(n, 6))
    E = rng.normal(size=(n, 20))
    return y, groups, T, F, E


def test_hybrid_fits_on_training_rows_only():
    y, g, T, F, E = data()
    X, layout = assemble(rowid=np.arange(len(y)), group=g, tile=T, feat=F, emb_dino=E)
    fn = lambda s: HybridClassifier(layout, use=("feat", "pod", "resid", "emb_dino"), r_pod=5, r_resid=3,
                                    pca_dim=4, base="lr", seed=s)
    P, classes, models = oof_proba(fn, X, y, g, seed=0, return_models=True)
    assert np.isfinite(P).all()
    for tr, te, m in models:
        assert set(m.seen_) == {"pod", "resid", "emb_dino", "clf"}
        log = [(k, ids, te) for k, ids in m.seen_.items()]
        assert_no_leak(log)
        for k, ids in m.seen_.items():
            assert ids <= set(tr.tolist()), k
        assert m.seen_["resid"] <= set(tr[y[tr] == "NORMAL"].tolist())   # healthy training eyes only


def test_fold_loops_fit_on_training_rows_only():
    y, g, T, _, _ = data()
    log = []
    pod_sweep_oof(T, y, g, 0, [2, 4], {"lr": lr}, log=log)
    residual_oof(T, y, g, 0, 3, log=log)
    assert len(log) == 10
    assert_no_leak(log)


def test_leak_check_catches_a_leak():
    y, g, T, _, _ = data()
    tr, te = folds(y, g, 0)[0]
    with pytest.raises(AssertionError):
        assert_no_leak([("pod", np.arange(len(y)), te)])   # fitted on everything -> must fail


def test_same_folds_as_v5():
    """v6 folds == retina.evaluation.oof_predictions folds (paired comparisons)."""
    from sklearn.model_selection import StratifiedGroupKFold
    y, g, T, _, _ = data()
    ref = list(StratifiedGroupKFold(5, shuffle=True, random_state=1).split(T, y, g))
    for (a, b), (c, d) in zip(folds(y, g, 1), ref):
        assert (a == c).all() and (b == d).all()


def test_image_pod_full_rank_reconstructs():
    rng = np.random.default_rng(1)
    T = rng.normal(size=(20, 50))
    pod = ImagePOD(19).fit(T)
    assert np.allclose(pod.reconstruct(T), T, atol=1e-6)
    assert pod.residual(T).max() < 1e-8
    assert np.all(np.diff(pod.energy_) >= -1e-12) and pod.energy_[-1] == pytest.approx(1.0, abs=1e-6)
