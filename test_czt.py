"""Test CZT package.

To run:

    pytest test_czt.py -v

"""

import numpy as np
import matplotlib.pyplot as plt 

import czt


def test_compare_different_czt_methods(debug=False):
    """Compare different CZT calculation methods."""
    
    print("Compare different CZT calculation methods")

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

    # Calculate CZT using different methods
    X_czt1 = czt.czt(x, simple=True)
    X_czt2 = czt.czt(x, t_method='ce')
    X_czt3 = czt.czt(x, t_method='pd')
    X_czt4 = czt.czt(x, t_method='mm')
    X_czt5 = czt.czt(x, t_method='scipy')
    X_czt6 = czt.czt(x, t_method='ce', f_method='fast')
    X_czt7 = czt.czt(x, t_method='pd', f_method='fast')

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Absolute value")
        plt.plot(np.abs(X_czt1), label="simple")
        plt.plot(np.abs(X_czt2), label="ce")
        plt.plot(np.abs(X_czt3), label="pd")
        plt.plot(np.abs(X_czt4), label="mm")
        plt.plot(np.abs(X_czt5), label="scipy")
        plt.plot(np.abs(X_czt6), label="ce / fast")
        plt.plot(np.abs(X_czt7), label="pd / fast")
        plt.legend()
        plt.figure()
        plt.title("Real component")
        plt.plot(X_czt1.real, label="simple")
        plt.plot(X_czt2.real, label="ce")
        plt.plot(X_czt3.real, label="pd")
        plt.plot(X_czt4.real, label="mm")
        plt.plot(X_czt5.real, label="scipy")
        plt.plot(X_czt6.real, label="ce / fast")
        plt.plot(X_czt7.real, label="pd / fast")
        plt.legend()
        plt.figure()
        plt.title("Imaginary component")
        plt.plot(X_czt1.imag, label="simple")
        plt.plot(X_czt2.imag, label="ce")
        plt.plot(X_czt3.imag, label="pd")
        plt.plot(X_czt4.imag, label="mm")
        plt.plot(X_czt5.imag, label="scipy")
        plt.plot(X_czt6.imag, label="ce / fast")
        plt.plot(X_czt7.imag, label="pd / fast")
        plt.legend()
        plt.show()

    # Compare Toeplitz matrix multiplication methods
    np.testing.assert_almost_equal(X_czt1, X_czt2, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt3, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt4, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt5, decimal=12)

    # Compare FFT methods
    np.testing.assert_allclose(X_czt1, X_czt6, atol=0.1)
    np.testing.assert_allclose(X_czt1, X_czt7, atol=0.1)


def test_compare_czt_fft_dft(debug=False):
    """Compare CZT, FFT and DFT."""
    
    print("Compare CZT, FFT and DFT")

    # Create time-domain data
    t = np.arange(0, 20e-3 + 1e-10, 1e-4)
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

    # CZT (defaults to FFT)
    X_czt = np.fft.fftshift(czt.czt(x))

    # FFT
    X_fft = np.fft.fftshift(np.fft.fft(x))

    # DFT
    _, X_dft = czt.dft(t, x)

    # Plot for debugging purposes
    if debug:
        plt.figure(figsize=(10, 8))
        plt.title("Absolute")
        plt.plot(np.abs(X_czt), label='CZT')
        plt.plot(np.abs(X_fft), label='FFT', ls='--')
        plt.plot(np.abs(X_dft), label='DFT', ls='--')
        plt.legend()
        plt.figure(figsize=(10, 8))
        plt.title("Real")
        plt.plot(X_czt.real, label='CZT')
        plt.plot(X_fft.real, label='FFT', ls='--')
        plt.plot(X_dft.real, label='DFT', ls='--')
        plt.legend()
        plt.figure(figsize=(10, 8))
        plt.title("Imaginary")
        plt.plot(X_czt.imag, label='CZT')
        plt.plot(X_fft.imag, label='FFT', ls='--')
        plt.plot(X_dft.imag, label='DFT', ls='--')
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(X_czt, X_fft, decimal=12)
    np.testing.assert_allclose(X_czt, X_dft, atol=0.2)


def test_czt_to_iczt(debug=False):
    """Test CZT -> ICZT."""
    
    print("Test CZT -> ICZT -> CZT")

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

    # CZT (defaults to FFT)
    X_czt = czt.czt(x)

    # ICZT
    x_iczt = czt.iczt(X_czt)
    x_iczt2 = czt.iczt(X_czt, simple=False)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Real")
        plt.plot(x.real)
        plt.plot(x_iczt.real)
        plt.plot(x_iczt2.real)
        plt.figure()
        plt.title("Imaginary")
        plt.plot(x.imag)
        plt.plot(x_iczt.imag)
        plt.plot(x_iczt2.imag)
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x, x_iczt, decimal=12)
    np.testing.assert_almost_equal(x, x_iczt2, decimal=12)


def test_time_to_freq_to_time(debug=False):
    """Test time -> frequency -> time domain conversions."""
    
    print("Test time -> freq -> time")

    # Create time-domain data
    t1 = np.arange(0, 20e-3, 1e-4)
    dt = t1[1] - t1[0]
    Fs = 1 / dt
    N = len(t1)

    # Signal
    def model(t):
        output = (1.0 * np.sin(2 * np.pi * 1e3 * t) + 
                  0.3 * np.sin(2 * np.pi * 2e3 * t) + 
                  0.1 * np.sin(2 * np.pi * 3e3 * t)) * np.exp(-1e3 * t)
        return output
    x1 = model(t1)

    # Frequency domain
    f, X = czt.time2freq(t1, x1)

    # Back to time domain
    t2, x2 = czt.freq2time(f, X)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Real")
        plt.plot(t1, x1.real, label='Original')
        plt.plot(t2, x2.real, label='Recovered', ls='--')
        plt.legend()
        plt.figure()
        plt.title("Imaginary")
        plt.plot(t1, x1.imag, label='Original')
        plt.plot(t2, x2.imag, label='Recovered', ls='--')
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_allclose(x1, x2, atol=0.01)


def test_compare_iczt_idft(debug=False):
    """Compare ICZT to IDFT."""
    
    print("Compare ICZT and IDFT")

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

    # Frequency domain using CZT
    f, X = czt.time2freq(t, x)

    # Get time-domain using ICZT
    _, x_iczt = czt.freq2time(f, X, t, t_orig=t)

    # Get time-domain using IDFT
    _, x_idft = czt.idft(f, X, t)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.plot(t, x.real, 'k', label="Original")
        plt.plot(t, x_iczt.real, 'g:', label="ICZT")
        plt.plot(t, x_idft.real, 'r--', label="IDFT")
        plt.legend()
        plt.figure()
        plt.plot(t, x.imag, 'k', label="Original")
        plt.plot(t, x_iczt.imag, 'g:', label="ICZT")
        plt.plot(t, x_idft.imag, 'r--', label="IDFT")
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x_iczt, x_idft, decimal=12)


def test_frequency_zoom(debug=False):
    """Test frequency zoom."""
    
    print("Test frequency zoom")

    # Create time-domain data
    t = np.arange(0, 20e-3 + 1e-10, 1e-4)
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

    # CZT
    f, X_czt1 = czt.time2freq(t, x)
    
    # DFT
    _, X_dft1 = czt.dft(t, x, f=f)

    # Truncate
    idx1, idx2 = 110, 180
    f_zoom = f[idx1:idx2]
    X_czt1, X_dft1 = X_czt1[idx1:idx2], X_dft1[idx1:idx2]
    
    # Zoom CZT
    _, X_czt2 = czt.time2freq(t, x, f_zoom, f_orig=f)
    
    # Zoom DFT
    _, X_dft2 = czt.dft(t, x, f=f_zoom)

    # Plot for debugging purposes
    if debug:
        plt.figure(figsize=(10, 8))
        plt.plot(f_zoom, np.abs(X_czt1), 'c', label='CZT')
        plt.plot(f_zoom, np.abs(X_dft1), 'k--', label='DFT')
        plt.plot(f_zoom, np.abs(X_czt2), 'r--', label='CZT (zoom)')
        plt.plot(f_zoom, np.abs(X_dft2), 'b:', label='DFT (zoom)')
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(X_czt1, X_czt2, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_dft1, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_dft2, decimal=12)


def test_compare_czt_to_analytic_expression(debug=False):
    """Compare CZT to analytic expression."""

    print("Compare CZT to analytic expression")

    # Create time-domain data
    t = np.arange(0, 50e-3 + 1e-10, 1e-5)
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

    # CZT
    f, X_czt = czt.time2freq(t, x)

    # Build frequency domain signal
    X1 = np.zeros_like(f, dtype=complex)
    idx = np.abs(f - 1e3).argmin()
    X1[idx] = 1 / 2j
    idx = np.abs(f + 1e3).argmin()
    X1[idx] = -1 / 2j
    idx = np.abs(f - 2e3).argmin()
    X1[idx] = 0.3 / 2j
    idx = np.abs(f + 2e3).argmin()
    X1[idx] = -0.3 / 2j
    idx = np.abs(f - 3e3).argmin()
    X1[idx] = 0.1 / 2j
    idx = np.abs(f + 3e3).argmin()
    X1[idx] = -0.1 / 2j
    X2 = 1 / (1e3 + 2j * np.pi * f)
    X = np.convolve(X1, X2)
    X = X[len(X)//4:-len(X)//4+1]
    X *= (f[1] - f[0]) * len(t)

    # Truncate
    mask = (0 < f) & (f < 5e3)
    f, X, X_czt = f[mask], X[mask], X_czt[mask]

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Real")
        plt.plot(f/1e3, X_czt.real)
        plt.plot(f/1e3, X.real, 'r--')
        plt.figure()
        plt.title("Imaginary")
        plt.plot(f/1e3, X_czt.imag)
        plt.plot(f/1e3, X.imag, 'r--')
        plt.show()

    # Compare
    np.testing.assert_allclose(X, X_czt, atol=0.02)


if __name__ == "__main__":

    test_compare_different_czt_methods(debug=True)
    test_compare_czt_fft_dft(debug=True)
    test_czt_to_iczt(debug=True)
    test_time_to_freq_to_time(debug=True)
    test_compare_iczt_idft(debug=True)
    test_frequency_zoom(debug=True)
    test_compare_czt_to_analytic_expression(debug=True)
