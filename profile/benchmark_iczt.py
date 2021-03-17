"""Benchmark czt.iczt function."""

import numpy as np
import czt
import perfplot


def model(t):
    output = (1.0 * np.sin(2 * np.pi * 1e3 * t) + 
              0.3 * np.sin(2 * np.pi * 2e3 * t) + 
              0.1 * np.sin(2 * np.pi * 3e3 * t)) * np.exp(-1e3 * t)
    return output


perfplot.show(
    setup=lambda n: czt.czt(model(np.linspace(0, 20e-3, n))),
    kernels=[
        lambda a: czt.iczt(a, simple=True),
        lambda a: czt.iczt(a, simple=False),
    ],
    labels=["simple=True", "simple=False"],
    n_range=[10 ** k for k in range(1, 8)],
    xlabel="Input length",
    # equality_check=np.allclose,
    equality_check=False,
    target_time_per_measurement=0.1,
)
