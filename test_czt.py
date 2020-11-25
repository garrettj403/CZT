import numpy as np 
import czt

def test_compare_czt_methods():
    
    # Time data
    t = np.arange(0, 20e-3, 1e-4)
    dt = t[1] - t[0]
    Fs = 1 / dt
    N = len(t)

    # Signal data
    def model(t):
        output = (1.0 * np.sin(2 * np.pi * 1e3 * t) + 
                  0.3 * np.sin(2 * np.pi * 2e3 * t) + 
                  0.1 * np.sin(2 * np.pi * 3e3 * t)) * np.exp(-1e3 * t)
        return output
    x = model(t)

    # Calculate CZT using different methods
    X_czt1 = czt.czt_simple(x)
    X_czt2 = czt.czt(x, t_method='ce')
    X_czt3 = czt.czt(x, t_method='pd')
    X_czt4 = czt.czt(x, t_method='mm')

    # Compare
    np.testing.assert_almost_equal(X_czt1, X_czt2, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt3, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt4, decimal=12)
