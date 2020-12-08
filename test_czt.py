"""Test CZT package.

To run:

    pytest test_czt.py -v

"""

import numpy as np

import czt


def test_compare_czt_methods(debug=False):
    """Compare different CZT calculation methods."""
    
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
    X_czt1 = czt.czt_simple(x)
    X_czt2 = czt.czt(x, t_method='ce')
    X_czt3 = czt.czt(x, t_method='pd')
    X_czt4 = czt.czt(x, t_method='mm')
    X_czt5 = czt.czt(x, t_method='ce', f_method='fast')
    X_czt6 = czt.czt(x, t_method='pd', f_method='fast')
    X_czt7 = czt.czt(x, t_method='mm', f_method='fast')

    # Debug
    if debug:
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.plot(np.abs(X_czt1))
        plt.plot(np.abs(X_czt2))
        plt.plot(np.abs(X_czt3))
        plt.plot(np.abs(X_czt4))
        plt.plot(np.abs(X_czt5))
        plt.plot(np.abs(X_czt6))
        plt.plot(np.abs(X_czt7))
        plt.figure()
        plt.plot(X_czt1.real)
        plt.plot(X_czt2.real)
        plt.plot(X_czt3.real)
        plt.plot(X_czt4.real)
        plt.plot(X_czt5.real)
        plt.plot(X_czt6.real)
        plt.plot(X_czt7.real)
        plt.figure()
        plt.plot(X_czt1.imag)
        plt.plot(X_czt2.imag)
        plt.plot(X_czt3.imag)
        plt.plot(X_czt4.imag)
        plt.plot(X_czt5.imag)
        plt.plot(X_czt6.imag)
        plt.plot(X_czt7.imag)
        plt.show()

    # Compare Toeplitz matrix multiplication methods
    np.testing.assert_almost_equal(X_czt1, X_czt2, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt3, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_czt4, decimal=12)

    # Compare FFT methods
    np.testing.assert_allclose(X_czt1, X_czt5, atol=0.1)
    np.testing.assert_allclose(X_czt1, X_czt6, atol=0.1)
    np.testing.assert_allclose(X_czt1, X_czt7, atol=0.1)


def test_compare_czt_fft_dft(debug=False):
    """Compare CZT to FFT and DFT."""
    
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

    # Debug
    if debug:
        import matplotlib.pyplot as plt 
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


def test_iczt(debug=False):
    """Test inverse CZT."""
    
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

    # Debug
    if debug:
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.plot(x.real)
        plt.plot(x_iczt.real)
        plt.figure()
        plt.plot(x.imag)
        plt.plot(x_iczt.imag)
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x, x_iczt, decimal=12)


def test_freq_to_time_convserions(debug=False):
    """Test frequency <-> time domain conversions."""
    
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

    # Debug
    if debug:
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.title("Absolute")
        plt.plot(t1, np.abs(x1), label='Original')
        plt.plot(t2, np.abs(x2), label='Recovered', ls='--')
        plt.legend()
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


def test_compare_czt_dft(debug=False):
    """Compare CZT to DFT."""
    
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
    f = np.linspace(0, 5e3, len(t))
    _, X_czt = czt.time2freq(t, x, f)

    # Frequency domain using CZT
    _, X_dft = czt.dft(t, x, f)

    # Debug
    if debug:
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.title("Absolute")
        plt.plot(f, np.abs(X_czt), 'k', label="CZT")
        plt.plot(f, np.abs(X_dft), 'r--', label="DFT")
        plt.legend()
        plt.figure()
        plt.title("Real")
        plt.plot(f, X_czt.real, 'k', label="CZT")
        plt.plot(f, X_dft.real, 'r--', label="DFT")
        plt.legend()
        plt.figure()
        plt.title("Imaginary")
        plt.plot(f, X_czt.imag, 'k', label="CZT")
        plt.plot(f, X_dft.imag, 'r--', label="DFT")
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_allclose(X_czt, X_dft, atol=0.2)


def test_compare_iczt_idft(debug=False):
    """Compare ICZT to IDFT."""
    
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
    _, x_iczt = czt.freq2time(f, X, t)

    # Get time-domain using IDFT
    _, x_idft = czt.idft(f, X, t)

    # Debug
    if debug:
        import matplotlib.pyplot as plt 
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
    # np.testing.assert_allclose(X_czt, X_dft, atol=0.2)


def test_frequency_zoom(debug=False):
    """Test frequency zoom."""
    
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
    f_czt1, X_czt1 = czt.time2freq(t, x)
    
    # DFT
    f_dft1, X_dft1 = czt.dft(t, x)

    # Truncate
    idx1, idx2 = 110, 180
    f_czt1, X_czt1 = f_czt1[idx1:idx2], X_czt1[idx1:idx2]
    f_dft1, X_dft1 = f_dft1[idx1:idx2], X_dft1[idx1:idx2]
    
    # Zoom CZT
    f_czt2, X_czt2 = czt.time2freq(t, x, f_czt1)
    
    # Zoom DFT
    f_dft2, X_dft2 = czt.dft(t, x, f_dft1)

    # Debug
    if debug:
        import matplotlib.pyplot as plt 
        plt.figure(figsize=(10, 8))
        plt.plot(f_czt1, np.abs(X_czt1))
        plt.plot(f_czt2, np.abs(X_czt2))
        plt.plot(f_dft1, np.abs(X_dft1))
        plt.plot(f_dft2, np.abs(X_dft2))
        plt.show()

    # All frequencies should be the same
    np.testing.assert_almost_equal(f_czt1, f_czt2, decimal=12)
    np.testing.assert_almost_equal(f_czt1, f_dft1, decimal=12)
    np.testing.assert_almost_equal(f_czt1, f_dft2, decimal=12)

    # Compare
    np.testing.assert_almost_equal(X_czt1, X_czt2, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_dft1, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_dft2, decimal=12)


if __name__ == "__main__":
    test_compare_czt_methods()
    test_compare_czt_fft_dft()
    test_iczt()
    test_freq_to_time_convserions()
    test_compare_czt_dft()
    test_compare_iczt_idft()
    test_frequency_zoom()
