"""Time czt.iczt function."""

import numpy as np
import czt
import timeit


def model(t):
    output = (1.0 * np.sin(2 * np.pi * 1e3 * t) + 
              0.3 * np.sin(2 * np.pi * 2e3 * t) + 
              0.1 * np.sin(2 * np.pi * 3e3 * t)) * np.exp(-1e3 * t)
    return output


# Create time-domain data
t = np.arange(0, 20e-3, 1e-4)
dt = t[1] - t[0]
Fs = 1 / dt
N = len(t)
x = model(t)

# CZT
X = czt.czt(x)

# Tests
def test1():
    czt.iczt(X, simple=True)
    return
def test2():
    czt.iczt(X, simple=False)
    return

N = 100
setup = "from __main__ import test1 as test"
print("Test 1: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "simple"))
N = 100
setup = "from __main__ import test2 as test"
print("Test 2: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "not simple"))
