"""Numerical parity: torch FID/IS/PR stack vs JAX fixtures (tests/parity/fixtures/fid.npz).

Runs in the torch env (no jax needed):
    pytest tests/parity/test_fid.py -v

Fixture is captured by:
    python tests/parity/capture_jax_fixtures.py --which fid

The first run downloads pytorch-fid's inception weights (~91 MB) into the
torch hub cache.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pt.utils.fid_util as fid_util  # noqa: E402
from pt.utils.torch_fid import resize  # noqa: E402
from pt.utils.torch_fid.fid import compute_frechet_distance  # noqa: E402
from pt.utils.torch_fid.precision_recall import compute_precision_recall  # noqa: E402

FIX = REPO / "tests" / "parity" / "fixtures" / "fid.npz"


@pytest.fixture(scope="module")
def fix():
    if not FIX.exists():
        pytest.skip(f"fixture missing: {FIX}")
    return dict(np.load(FIX))


@pytest.fixture(scope="module")
def inception_model():
    from pt.utils.torch_fid.inception import InceptionV3

    return InceptionV3().eval()  # CPU for determinism


@pytest.fixture(scope="module")
def torch_outputs(fix, inception_model):
    """(pooled, logits) from the torch inception on the fixture's resized input."""
    x = torch.from_numpy(fix["resized"])
    pooled_list, logits_list = [], []
    with torch.no_grad():
        for i in range(0, len(x), 16):
            pooled, logits = inception_model(x[i : i + 16])
            pooled_list.append(pooled.numpy())
            logits_list.append(logits.numpy())
    return np.concatenate(pooled_list), np.concatenate(logits_list)


def _stats(feats):
    f64 = feats.astype(np.float64)
    return np.mean(f64, axis=0), np.cov(f64, rowvar=False)


def test_resize_matches_fixture(fix):
    x = torch.from_numpy(fix["images"].astype(np.float32).transpose(0, 3, 1, 2))
    out = resize.forward(x).numpy()
    diff = np.abs(out - fix["resized"]).max()
    assert diff <= 1e-5, f"resize max abs diff {diff}"


def test_pooled_features_match(fix, torch_outputs):
    pooled, _ = torch_outputs
    diff = np.abs(pooled - fix["pooled"]).max()
    assert diff <= 1e-3, f"pooled features max abs diff {diff}"


def test_logits_match(fix, torch_outputs):
    _, logits = torch_outputs
    diff = np.abs(logits - fix["logits"]).max()
    assert diff <= 1e-3, f"logits max abs diff {diff}"


def test_fid_function_exact_on_fixture_features(fix):
    """fid.py is a verbatim numpy copy: same inputs must give the same value.

    Not bit-exact across environments: np.linalg.eigvals on the (rank-deficient)
    2048x2048 sigma product depends on the numpy/LAPACK build (jax env ships
    numpy 1.26, torch env numpy 2.2; observed diff ~2e-7).
    """
    a_mu, a_sigma = _stats(fix["pooled"][:32])
    b_mu, b_sigma = _stats(fix["pooled"][32:])
    val = compute_frechet_distance(a_mu, b_mu, a_sigma, b_sigma)
    assert abs(val - float(fix["fid_half"])) <= 1e-5, (val, float(fix["fid_half"]))


def test_fid_from_torch_features(fix, torch_outputs):
    pooled, _ = torch_outputs
    a_mu, a_sigma = _stats(pooled[:32])
    b_mu, b_sigma = _stats(pooled[32:])
    val = compute_frechet_distance(a_mu, b_mu, a_sigma, b_sigma)
    assert abs(val - float(fix["fid_half"])) <= 0.05, (val, float(fix["fid_half"]))


def test_inception_score_on_fixture_logits(fix):
    mean, std = fid_util._compute_inception_score(fix["logits"])
    ref_mean, ref_std = float(fix["isc_mean"]), float(fix["isc_std"])
    assert abs(mean - ref_mean) <= 1e-3 * ref_mean, (mean, ref_mean)
    assert abs(std - ref_std) <= 1e-3 * max(ref_std, 1e-6), (std, ref_std)


def test_inception_score_on_torch_logits(fix, torch_outputs):
    _, logits = torch_outputs
    mean, _ = fid_util._compute_inception_score(logits)
    ref_mean = float(fix["isc_mean"])
    assert abs(mean - ref_mean) <= 1e-3 * ref_mean, (mean, ref_mean)


def test_precision_recall_on_fixture_features(fix):
    precision, recall = compute_precision_recall(
        fix["pooled"][:32].astype(np.float64), fix["pooled"][32:].astype(np.float64), k=3
    )
    assert abs(float(precision) - float(fix["precision"])) <= 0.02
    assert abs(float(recall) - float(fix["recall"])) <= 0.02


def test_precision_recall_on_torch_features(fix, torch_outputs):
    pooled, _ = torch_outputs
    precision, recall = compute_precision_recall(
        pooled[:32].astype(np.float64), pooled[32:].astype(np.float64), k=3
    )
    assert abs(float(precision) - float(fix["precision"])) <= 0.02
    assert abs(float(recall) - float(fix["recall"])) <= 0.02


class _DummyLogger:
    def __init__(self):
        self.dicts = []
        self.images = []

    def log_dict(self, d):
        self.dicts.append(d)

    def log_image(self, key, value):
        self.images.append((key, np.asarray(value).shape))


def test_evaluate_fid_end_to_end(fix, tmp_path, monkeypatch):
    """evaluate_fid smoke test: ref stats = first half, gen_func = second half.

    Exercises the padding/mask path (last batch is short) and must reproduce
    the fixture's half-vs-half FID.
    """
    ref_mu, ref_sigma = _stats(fix["pooled"][:32])
    ref_path = tmp_path / "ref_stats.npz"
    np.savez(ref_path, mu=ref_mu, sigma=ref_sigma)
    monkeypatch.setitem(fid_util._DATASET_STATS, "imagenet256", str(ref_path))

    # (x + 0.5) / 255 survives _to_uint8's truncation exactly.
    imgs = (fix["images"][32:].astype(np.float32) + 0.5) / 255.0

    # 3 batches: 12 + 12 + 8; the last is zero-padded to 12 inside evaluate_fid.
    # Batch leaves must share the batch dim (goal_bsz comes from the first leaf).
    def _batch(n):
        return (np.zeros((n, 4, 4, 3), np.float32), np.arange(n))

    loader = [_batch(12), _batch(12), _batch(8)]

    rngs = []

    def gen_func(batch, *, rng):
        _, labels = batch
        b = labels.shape[0]
        i0 = 12 * len(rngs)
        rngs.append(rng)
        chunk = imgs[i0 : i0 + b]
        if len(chunk) < b:  # padded entries: return garbage, mask drops them
            pad = np.zeros((b - len(chunk), *chunk.shape[1:]), chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    logger = _DummyLogger()
    metrics = fid_util.evaluate_fid(
        dataset_name="imagenet256",
        gen_func=gen_func,
        gen_params={},
        eval_loader=loader,
        logger=logger,
        num_samples=32,
        log_folder="fid",
        log_prefix="parity",
        eval_prc_recall=False,
        eval_isc=True,
        eval_fid=True,
        rng_eval=0,
    )

    assert rngs == [0, 1, 2]
    assert abs(metrics["fid"] - float(fix["fid_half"])) <= 0.05, metrics
    assert np.isfinite(metrics["isc_mean"]) and metrics["isc_mean"] > 0
    assert "fid_time" in metrics
    assert len(logger.dicts) == 1 and any("parity_fid" in k for k in logger.dicts[0])
    assert len(logger.images) == 1
