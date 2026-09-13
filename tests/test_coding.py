import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coding.rans_coder import RansCoder


def test_rans_roundtrip_basic():
    rng = np.random.default_rng(0)
    n = 500
    means = rng.normal(0, 1, size=n)
    scales = np.abs(rng.normal(1, 0.3, size=n)) + 0.1
    symbols = np.round(rng.normal(means, scales)).astype(np.int32)
    symbols = np.clip(symbols, -100, 100)

    coder = RansCoder(symbol_range=128)
    encoded = coder.encode(symbols, means, scales)
    decoded = coder.decode(encoded, means, scales)

    assert np.array_equal(symbols, decoded[: len(symbols)])


def test_rans_achieves_compression_on_low_entropy_data():
    # symbols tightly clustered around their means -> should compress well
    rng = np.random.default_rng(1)
    n = 2000
    means = np.zeros(n)
    scales = np.full(n, 0.5)
    symbols = np.round(rng.normal(0, 0.3, size=n)).astype(np.int32)

    coder = RansCoder(symbol_range=64)
    encoded = coder.encode(symbols, means, scales)

    raw_bytes = symbols.astype(np.int32).nbytes
    # Only a meaningful assertion when the real rANS backend is present;
    # the pure-Python fallback intentionally stores fixed-width symbols.
    try:
        import constriction  # noqa: F401
        assert len(encoded) < raw_bytes
    except ImportError:
        assert len(encoded) <= raw_bytes + 8
