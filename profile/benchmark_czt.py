"""Benchmark czt.czt function."""

import numpy as np
import czt
import perfplot


def model(t):
    """Signal model."""
    output = (1.0 * np.sin(2 * np.pi * 1e3 * t) +
              0.3 * np.sin(2 * np.pi * 2e3 * t) +
              0.1 * np.sin(2 * np.pi * 3e3 * t)) * np.exp(-1e3 * t)
    return output


perfplot.show(
    setup=lambda n: model(np.linspace(0, 20e-3, n)),
    kernels=[
        # lambda a: czt.czt(a, simple=True),
        lambda a: czt.czt(a, t_method='ce'),
        lambda a: czt.czt(a, t_method='pd'),
        # lambda a: czt.czt(a, t_method='mm'),
        lambda a: czt.czt(a, t_method='scipy'),
        # lambda a: czt.czt(a, t_method='ce', f_method='recursive'),
        # lambda a: czt.czt(a, t_method='pd', f_method='recursive'),
    ],
    # labels=["simple", "ce", "pd", "mm", "scipy", "ce/recursive", "pd/recursive"],
    labels=["ce", "pd", "scipy"],
    n_range=[10 ** k for k in range(1, 7)],
    xlabel="Input length",
    equality_check=np.allclose,
    target_time_per_measurement=0.1,
)
