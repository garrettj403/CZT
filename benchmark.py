"""Benchmark czt.czt function."""

import timeit
import time
import numpy as np
import czt


# Create time-domain data
t = np.arange(0, 20e-3, 1e-4)
dt = t[1] - t[0]
Fs = 1 / dt
N = len(t)

# Signal
def model(t):
    output = (1.0 * np.sin(2 * np.pi * 1e3 * t) + 
              0.3 * np.sin(2 * np.pi * 2e3 * t) + 
              0.1 * np.sin(2 * np.pi * 3e3 * t)) * np.exp(-1e3 * t)
    return output
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
# def test6():
#     czt.czt(x, t_method='ce', f_method='fast')
#     return
# def test7():
#     czt.czt(x, t_method='pd', f_method='fast')
#     return

N = 100
setup = "from __main__ import test1 as test"
print("Test 1: ", timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, " ms")
setup = "from __main__ import test2 as test"
print("Test 2: ", timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, " ms")
setup = "from __main__ import test3 as test"
print("Test 3: ", timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, " ms")
setup = "from __main__ import test4 as test"
print("Test 4: ", timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, " ms")
setup = "from __main__ import test5 as test"
print("Test 5: ", timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, " ms")
# setup = "from __main__ import test6 as test"
# print("Test 6: ", timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, " ms")
# setup = "from __main__ import test7 as test"
# print("Test 7: ", timeit.Timer("test()", setup=setup).timeit(number=N)/N*1000, " ms")
