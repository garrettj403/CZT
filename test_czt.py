"""Test CZT package.

To run:

    pytest test_czt.py -v

"""

import numpy as np
import matplotlib.pyplot as plt 

import czt


def test_compare_different_czt_methods(debug=False):
    print("Compare different CZT calculation methods")

    # Create time-domain data
    t = np.arange(0, 20e-3, 1e-4)

    # Signal
    x = _signal_model(t)

    # Calculate CZT using different methods
    X_czt1 = czt.czt(x, simple=True)
    X_czt2 = czt.czt(x, t_method='ce')
    X_czt3 = czt.czt(x, t_method='pd')
    X_czt4 = czt.czt(x, t_method='mm')
    X_czt5 = czt.czt(x, t_method='scipy')
    X_czt6 = czt.czt(x, t_method='ce', f_method='recursive')
    X_czt7 = czt.czt(x, t_method='pd', f_method='recursive')

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Imaginary component")
        plt.plot(X_czt1.imag, label="simple")
        plt.plot(X_czt2.imag, label="ce")
        plt.plot(X_czt3.imag, label="pd")
        plt.plot(X_czt4.imag, label="mm")
        plt.plot(X_czt5.imag, label="scipy")
        plt.plot(X_czt6.imag, label="ce / recursive")
        plt.plot(X_czt7.imag, label="pd / recursive")
        plt.legend()
        plt.figure()
        plt.title("Real component")
        plt.plot(X_czt1.real, label="simple")
        plt.plot(X_czt2.real, label="ce")
        plt.plot(X_czt3.real, label="pd")
        plt.plot(X_czt4.real, label="mm")
        plt.plot(X_czt5.real, label="scipy")
        plt.plot(X_czt6.real, label="ce / recursive")
        plt.plot(X_czt7.real, label="pd / recursive")
        plt.legend()
        plt.figure()
        plt.title("Absolute value")
        plt.plot(np.abs(X_czt1), label="simple")
        plt.plot(np.abs(X_czt2), label="ce")
        plt.plot(np.abs(X_czt3), label="pd")
        plt.plot(np.abs(X_czt4), label="mm")
        plt.plot(np.abs(X_czt5), label="scipy")
        plt.plot(np.abs(X_czt6), label="ce / recursive")
        plt.plot(np.abs(X_czt7), label="pd / recursive")
        plt.legend()
        plt.show()

    # Compare Toeplitz matrix multiplication methods
    np.testing.assert_almost_equal(X_czt1, X_czt2, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt3, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt4, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt5, decimal=12)

    # Compare FFT methods
    np.testing.assert_almost_equal(X_czt1, X_czt6, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt7, decimal=12)


def test_compare_czt_fft_dft(debug=False):
    print("Compare CZT, FFT and DFT")

    # Create time-domain data
    t = np.arange(0, 20e-3 + 1e-10, 1e-4)
    dt = t[1] - t[0]
    fs = 1 / dt

    # Frequency sweep
    f = np.fft.fftshift(np.fft.fftfreq(len(t)) * fs)

    # Signal
    x = _signal_model(t)

    # CZT (defaults to FFT settings)
    X_czt = np.fft.fftshift(czt.czt(x))

    # FFT
    X_fft = np.fft.fftshift(np.fft.fft(x))

    # DFT (defaults to FFT settings)
    _, X_dft = czt.dft(t, x)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Imaginary")
        plt.plot(f, X_czt.imag, label='CZT')
        plt.plot(f, X_fft.imag, label='FFT', ls='--')
        plt.plot(f, X_dft.imag, label='DFT', ls='--')
        plt.legend()
        plt.figure()
        plt.title("Real")
        plt.plot(f, X_czt.real, label='CZT')
        plt.plot(f, X_fft.real, label='FFT', ls='--')
        plt.plot(f, X_dft.real, label='DFT', ls='--')
        plt.legend()
        plt.figure()
        plt.title("Absolute")
        plt.plot(f, np.abs(X_czt), label='CZT')
        plt.plot(f, np.abs(X_fft), label='FFT', ls='--')
        plt.plot(f, np.abs(X_dft), label='DFT', ls='--')
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(X_czt, X_fft, decimal=12)
    np.testing.assert_almost_equal(X_czt, X_dft, decimal=12)


def test_czt_to_iczt(debug=False):
    print("Test CZT -> ICZT")

    # Create time-domain data
    t = np.arange(0, 20e-3, 1e-4)

    # Signal
    x = _signal_model(t)

    # CZT (defaults to FFT)
    X_czt = czt.czt(x)

    # ICZT
    x_iczt1 = czt.iczt(X_czt)
    x_iczt2 = czt.iczt(X_czt, simple=False)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Imaginary")
        plt.plot(t*1e3, x.imag)
        plt.plot(t*1e3, x_iczt1.imag)
        plt.plot(t*1e3, x_iczt2.imag)
        plt.figure()
        plt.title("Real")
        plt.plot(t*1e3, x.real)
        plt.plot(t*1e3, x_iczt1.real)
        plt.plot(t*1e3, x_iczt2.real)
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x, x_iczt1, decimal=12)
    np.testing.assert_almost_equal(x, x_iczt2, decimal=12)


def test_time_to_freq_to_time(debug=False):
    print("Test time -> freq -> time")

    # Create time-domain data
    t1 = np.arange(0, 20e-3, 1e-4)
    x1 = _signal_model(t1)

    # Frequency domain
    f, X = czt.time2freq(t1, x1)

    # Back to time domain
    t2, x2 = czt.freq2time(f, X, t=t1)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Imaginary")
        plt.plot(t1, x1.imag, 'k', label='Original')
        plt.plot(t2, x2.imag, 'r', label='Recovered')
        plt.legend()
        plt.figure()
        plt.title("Real")
        plt.plot(t1, x1.real, 'k', label='Original')
        plt.plot(t2, x2.real, 'r', label='Recovered')
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x1, x2, decimal=12)


def test_compare_iczt_idft(debug=False):
    print("Compare ICZT and IDFT")

    # Create time-domain data
    t = np.arange(0, 20e-3, 1e-4)

    # Signal
    x = _signal_model(t)

    # Frequency domain using DFT
    f, X = czt.dft(t, x)

    # Get time-domain using ICZT
    _, x_iczt = czt.freq2time(f, X, t)

    # Get time-domain using IDFT
    _, x_idft = czt.idft(f, X, t)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Imaginary")
        plt.plot(t, x.imag, 'k', label="Original")
        plt.plot(t, x_iczt.imag, 'g:', label="ICZT")
        plt.plot(t, x_idft.imag, 'r--', label="IDFT")
        plt.legend()
        plt.figure()
        plt.title("Real")
        plt.plot(t, x.real, 'k', label="Original")
        plt.plot(t, x_iczt.real, 'g:', label="ICZT")
        plt.plot(t, x_idft.real, 'r--', label="IDFT")
        plt.legend()
        plt.figure()
        plt.title("Real: error")
        plt.plot(t, x_iczt.real - x.real, 'k', label="Original")
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x_iczt, x, decimal=12)
    np.testing.assert_almost_equal(x_idft, x, decimal=12)
    np.testing.assert_almost_equal(x_iczt, x_idft, decimal=12)


def test_frequency_zoom(debug=False):
    print("Test frequency zoom")

    # Create time-domain data
    t = np.arange(0, 20e-3 + 1e-10, 1e-4)
    dt = t[1] - t[0]

    # Signal
    x = _signal_model(t)

    # Standard FFT frequency range
    f = np.fft.fftshift(np.fft.fftfreq(len(t), dt))

    # DFT
    f, X_dft1 = czt.dft(t, x, f=f)

    # CZT
    f, X_czt1 = czt.time2freq(t, x, f=f)

    # Truncate
    idx1, idx2 = 110, 180
    f_zoom = f[idx1:idx2]
    X_czt1, X_dft1 = X_czt1[idx1:idx2], X_dft1[idx1:idx2]
    
    # Zoom DFT
    _, X_dft2 = czt.dft(t, x, f_zoom)

    # Zoom CZT
    _, X_czt2 = czt.time2freq(t, x, f_zoom)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Imaginary")
        plt.plot(f_zoom, np.imag(X_czt1), 'c', label='CZT')
        plt.plot(f_zoom, np.imag(X_dft1), 'k--', label='DFT')
        plt.plot(f_zoom, np.imag(X_czt2), 'r--', label='CZT (zoom)')
        plt.plot(f_zoom, np.imag(X_dft2), 'b:', label='DFT (zoom)')
        plt.legend()
        plt.figure()
        plt.title("Real")
        plt.plot(f_zoom, np.real(X_czt1), 'c', label='CZT')
        plt.plot(f_zoom, np.real(X_dft1), 'k--', label='DFT')
        plt.plot(f_zoom, np.real(X_czt2), 'r--', label='CZT (zoom)')
        plt.plot(f_zoom, np.real(X_dft2), 'b:', label='DFT (zoom)')
        plt.legend()
        plt.figure()
        plt.title("Absolute")
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

def test_time_zoom(debug=False):
    print("Test time zoom")

    # Create time-domain data
    t = np.arange(0, 20e-3 + 1e-10, 1e-4)
    dt = t[1] - t[0]

    # Signal
    x = _signal_model(t)

    # Standard FFT frequency range
    f = np.fft.fftshift(np.fft.fftfreq(len(t), dt))

    # DFT
    f, X = czt.dft(t, x, f=f)

    # Time domain
    t1, x1 = czt.freq2time(f, X, t=t)

    # Time domain: zoom
    t2 = t1[(0.001 <= t1) & (t1 <= 0.002)]
    _, x2 = czt.freq2time(f, X, t=t2)

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Imaginary")
        plt.plot(t, np.imag(x), 'c', label='Original')
        plt.plot(t1, np.imag(x1), 'k:', label='freq2time: full')
        plt.plot(t2, np.imag(x2), 'r--', label='freq2time: full')
        plt.xlim([0, 0.003])
        plt.legend()
        plt.figure()
        plt.title("Real")
        plt.plot(t, np.real(x), 'c', label='Original')
        plt.plot(t1, np.real(x1), 'k:', label='freq2time: full')
        plt.plot(t2, np.real(x2), 'r--', label='freq2time: full')
        plt.xlim([0, 0.003])
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x, x1, decimal=12)
    np.testing.assert_almost_equal(x[(0.001 <= t1) & (t1 <= 0.002)], x2, decimal=12)


def test_compare_czt_to_analytic_expression(debug=False):
    print("Compare CZT to analytic expression")

    # Create time-domain data
    t = np.linspace(0, 50, 10001) * 1e-3

    # Signal
    x = _signal_model(t)

    # CZT
    f, X_czt = czt.time2freq(t, x)

    # Build frequency domain signal
    X = _signal_model_f(f, len(t))

    # Transform back to time-domain
    _, x_iczt = czt.freq2time(f, X_czt, t=t)

    # Truncate
    mask = (0 < f) & (f < 5e3)
    f, X, X_czt = f[mask], X[mask], X_czt[mask]

    # Plot for debugging purposes
    if debug:
        plt.figure()
        plt.title("Freq-Domain: Imaginary")
        plt.plot(f/1e3, X_czt.imag, label="CZT")
        plt.plot(f/1e3, X.imag, 'r--', label="Analytic")
        plt.legend()
        plt.figure()
        plt.title("Freq-Domain: Real")
        plt.plot(f / 1e3, X_czt.real, label="CZT")
        plt.plot(f / 1e3, X.real, 'r--', label="Analytic")
        plt.legend()
        plt.figure()
        plt.title("Freq-Domain: Absolute")
        plt.plot(f / 1e3, np.abs(X_czt), label="CZT")
        plt.plot(f / 1e3, np.abs(X), 'r--', label="Analytic")
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_allclose(X, X_czt, atol=0.1)
    np.testing.assert_almost_equal(x, x_iczt, decimal=12)


def _signal_model(tt):
    """Generate time-domain signal for tests."""

    output = (1.0 * np.sin(2 * np.pi * 1e3 * tt) +
              0.3 * np.sin(2 * np.pi * 2e3 * tt) +
              0.1 * np.sin(2 * np.pi * 3e3 * tt)) * np.exp(-1e3 * tt)

    return output


def _signal_model_f(ff, t_npts):
    """Generate frequency-domain signal for tests."""

    X1 = np.zeros_like(ff, dtype=complex)
    idx = np.abs(ff - 1e3).argmin()
    X1[idx] = 1 / 2j
    idx = np.abs(ff + 1e3).argmin()
    X1[idx] = -1 / 2j
    idx = np.abs(ff - 2e3).argmin()
    X1[idx] = 0.3 / 2j
    idx = np.abs(ff + 2e3).argmin()
    X1[idx] = -0.3 / 2j
    idx = np.abs(ff - 3e3).argmin()
    X1[idx] = 0.1 / 2j
    idx = np.abs(ff + 3e3).argmin()
    X1[idx] = -0.1 / 2j
    
    X2 = 1 / (1e3 + 2j * np.pi * ff)

    X = np.convolve(X1, X2)
    X = X[len(X) // 4:-len(X) // 4 + 1]

    X *= (ff[1] - ff[0]) * t_npts

    return X


if __name__ == "__main__":

    # test_compare_different_czt_methods(debug=True)
    # test_compare_czt_fft_dft(debug=True)
    # test_czt_to_iczt(debug=True)
    # test_time_to_freq_to_time(debug=True)
    # test_compare_iczt_idft(debug=True)
    # test_frequency_zoom(debug=True)
    test_time_zoom(debug=True)
    # test_compare_czt_to_analytic_expression(debug=True)
