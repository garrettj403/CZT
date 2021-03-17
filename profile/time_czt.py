"""Time czt.czt function."""

import numpy as np
import czt
import timeit


def model(t):
    """Signal model."""
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


# Tests
def test1():
    czt.czt(x, simple=True)
    return
def test2():
    czt.czt(x, t_method='ce')
    return
def test3():
    czt.czt(x, t_method='pd')
    return
def test4():
    czt.czt(x, t_method='mm')
    return
def test5():
    czt.czt(x, t_method='scipy')
    return
def test6():
    czt.czt(x, t_method='ce', f_method='recursive')
    return
def test7():
    czt.czt(x, t_method='pd', f_method='recursive')
    return


N = 100
setup = "from __main__ import test1 as test"
print("Test 1: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "simple"))
N = 1000
setup = "from __main__ import test2 as test"
print("Test 2: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "ce"))
N = 100
setup = "from __main__ import test3 as test"
print("Test 3: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "pd"))
N = 1000
setup = "from __main__ import test4 as test"
print("Test 4: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "mm"))
N = 1000
setup = "from __main__ import test5 as test"
print("Test 5: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "scipy"))
N = 10
setup = "from __main__ import test6 as test"
print("Test 6: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "ce / recursive"))
N = 10
setup = "from __main__ import test7 as test"
print("Test 7: {:7.4f} ms\t{}".format(timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, "pd / recursive"))
