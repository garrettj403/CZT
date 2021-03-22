"""Calculate the Chirp Z-transform (CZT).

CZT reference:

   Lawrence R. Rabiner, Ronald W. Schafer, and Charles M. Rader, "The chirp 
   z-transform algorithm and its application," Bell Syst. Tech. J. 48, 
   1249-1292 (1969).

CZT computation reference:

    Sukhoy, V., Stoytchev, A. "Generalizing the inverse FFT off the unit 
    circle," Sci Rep 9, 14443 (2019). 

"""

import numpy as np
from scipy.linalg import toeplitz, matmul_toeplitz


# CZT ------------------------------------------------------------------------

def czt(x, M=None, W=None, A=1.0, simple=False, t_method="ce", f_method="numpy"):
    """Calculate the Chirp Z-transform (CZT).

    Solves in O(n log n) time.

    See algorithm 1 in Sukhoy & Stoytchev 2019.

    Args:
        x (np.ndarray): input array
        M (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point
        simple (bool): use simple algorithm? (very slow)
        t_method (str): Toeplitz matrix multiplication method. 'ce' for
            circulant embedding, 'pd' for Pustylnikov's decomposition, 'mm'
            for simple matrix multiplication, 'scipy' for matmul_toeplitz
            from scipy.linalg.
        f_method (str): FFT method. 'numpy' for FFT from NumPy,
        'recursive' for recursive method.

    Returns:
        np.ndarray: Chirp Z-transform

    """

    # Unpack arguments
    N = len(x)
    if M is None:
        M = N
    if W is None:
        W = np.exp(-2j * np.pi / M)
    A = np.complex128(A)
    W = np.complex128(W)

    # Simple algorithm (very slow)
    if simple:
        k = np.arange(M)
        X = np.zeros(M, dtype=complex)
        z = A * W ** -k
        for n in range(N):
            X += x[n] * z ** -n
        return X

    # Algorithm 1 from Sukhoy & Stoytchev 2019
    k = np.arange(max(M, N))
    Wk22 = W ** (-(k ** 2) / 2)
    r = Wk22[:N]
    c = Wk22[:M]
    X = A ** -k[:N] * x / r
    try:
        toeplitz_mult = _available_t_methods[t_method]  # now this raises an key error
    except KeyError:
        raise ValueError(f"t_method {t_method} not recognized. Must be one of {list(_available_t_methods.keys())}")
    X = toeplitz_mult(r, c, X, f_method)
    return X / c


def iczt(X, N=None, W=None, A=1.0, simple=True, t_method="scipy", f_method="numpy"):
    """Calculate inverse Chirp Z-transform (ICZT).

    Uses an efficient algorithm. Solves in O(n log n) time.

    See algorithm 2 in Sukhoy & Stoytchev 2019.

    Args:
        X (np.ndarray): input array
        N (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point
        simple (bool): calculate ICZT using simple method (using CZT and
            conjugate)
        t_method (str): Toeplitz matrix multiplication method. 'ce' for
            circulant embedding, 'pd' for Pustylnikov's decomposition, 'mm'
            for simple matrix multiplication, 'scipy' for matmul_toeplitz
            from scipy.linalg. Ignored if you are not using the simple ICZT
            method.
        f_method (str): FFT method. 'numpy' for FFT from NumPy,
        'recursive' for recursive method.

    Returns:
        np.ndarray: Inverse Chirp Z-transform

    """

    # Unpack arguments
    M = len(X)
    if N is None:
        N = M
    if W is None:
        W = np.exp(-2j * np.pi / M)
    A = np.complex128(A)
    W = np.complex128(W)

    # Simple algorithm
    if simple:
        return np.conj(czt(np.conj(X), M=N, W=W, A=A, t_method=t_method, f_method=f_method)) / M

    # Algorithm 2 from Sukhoy & Stoytchev 2019
    if M != N:
        raise ValueError("M must be equal to N")
    n = N
    k = np.arange(n)
    Wk22 = W ** (-(k ** 2) / 2)
    x = Wk22 * X
    p = np.r_[1, (W ** k[1:] - 1).cumprod()]
    u = (-1) ** k * W ** (k * (k - n + 0.5) + (n / 2 - 0.5) * n) / p
    # equivalent to:
    # u = (-1) ** k * W ** ((2 * k ** 2 - (2 * n - 1) * k + n * (n - 1)) / 2) / p
    u /= p[::-1]
    z = np.zeros(n, dtype=complex)
    uhat = np.r_[0, u[-1:0:-1]]
    util = np.r_[u[0], np.zeros(n - 1)]
    try:
        toeplitz_mult = _available_t_methods[t_method]  # now this raises an key error
    except KeyError:
        raise ValueError(f"t_method {t_method} not recognized. Must be one of {list(_available_t_methods.keys())}")
    # Note: there is difference in accuracy here depending on the method. Have to check.
    x1 = toeplitz_mult(uhat, z, x, f_method)
    x1 = toeplitz_mult(z, uhat, x1, f_method)
    x2 = toeplitz_mult(u, util, x, f_method)
    x2 = toeplitz_mult(util, u, x2, f_method)
    x = (x2 - x1) / u[0]
    x = A ** k * Wk22 * x
    return x


# OTHER TRANSFORMS -----------------------------------------------------------

def dft(t, x, f=None):
    """Transform signal from time- to frequency-domain using a Discrete
    Fourier Transform (DFT).

    Used for testing CZT algorithm.

    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal

    Returns:
        np.ndarray: frequency-domain signal

    """

    if f is None:
        # Default to FFT frequency sweep
        nt = len(t)
        tspan = t[-1] - t[0]
        dt = tspan / (nt - 1)  # more accurate than t[1] - t[0]
        f = np.fft.fftshift(np.fft.fftfreq(nt, dt))

    X = np.empty(len(f), dtype=complex)
    for k in range(len(f)):
        X[k] = np.sum(x * np.exp(-2j * np.pi * f[k] * t))

    return f, X


def idft(f, X, t=None):
    """Transform signal from time- to frequency-domain using an Inverse
    Discrete Fourier Transform (IDFT).

    Used for testing the ICZT algorithm.

    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal

    Returns:
        np.ndarray: time-domain signal

    """

    if t is None:
        # Default to FFT time sweep
        nf = len(f)
        fspan = f[-1] - f[0]
        df = fspan / (nf - 1)  # more accurate than f[1] - f[0]
        t = np.fft.fftshift(np.fft.fftfreq(nf, df))

    x = np.empty(len(t), dtype=complex)
    for n in range(len(t)):
        x[n] = np.sum(X * np.exp(2j * np.pi * f * t[n]))
    x /= len(t)

    return t, x


# FREQ <--> TIME-DOMAIN CONVERSION -------------------------------------------

def time2freq(t, x, f=None):
    """Transform a time-domain signal to the frequency-domain.

    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal, optional, defaults to
            standard FFT frequency sweep

    Returns:
        frequency-domain signal

    """

    # Input time array
    nt = len(t)
    tspan = t[-1] - t[0]
    dt = tspan / (nt - 1)  # more accurate than t[1] - t[0]

    # Output frequency array
    if f is None:
        # Default to FFT frequency sweep
        f = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    else:
        f = np.copy(f)
    nf = len(f)
    fspan = f[-1] - f[0]
    df = fspan / (nf - 1)  # more accurate than f[1] - f[0]

    # Starting point
    A = np.exp(2j * np.pi * f[0] * dt)

    # Step
    W = np.exp(-2j * np.pi * df * dt)

    # Phase correction
    phase = np.exp(-2j * np.pi * t[0] * f)

    # Frequency-domain transform
    freq_data = czt(x, nf, W, A) * phase

    return f, freq_data


def freq2time(f, X, t=None):
    """Transform a frequency-domain signal to the time-domain.

    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal, optional, defaults to standard
            FFT time sweep

    Returns:
        time-domain signal

    """

    # Input frequency array
    nf = len(f)
    fspan = f[-1] - f[0]
    df = fspan / (nf - 1)  # more accurate than f[1] - f[0]

    # Output time array
    if t is None:
        # Default to FFT time sweep
        t = np.fft.fftshift(np.fft.fftfreq(nf, df))
    else:
        t = np.copy(t)
    nt = len(t)
    tspan = t[-1] - t[0]
    dt = tspan / (nt - 1)  # more accurate than t[1] - t[0]

    # Starting point
    A = np.exp(-2j * np.pi * t[0] * df)

    # Step
    W = np.exp(2j * np.pi * df * dt)

    # Phase correction
    phase = np.exp(2j * np.pi * f[0] * t)

    # Time-domain transform
    time_data = czt(X, nt, W=W, A=A) * phase / nf

    return t, time_data


# HELPER FUNCTIONS -----------------------------------------------------------

def _toeplitz_mult_ce(r, c, x, f_method="numpy"):
    """Multiply Toeplitz matrix by vector using circulant embedding.

    See algorithm S1 in Sukhoy & Stoytchev 2019:

       Compute the product y = Tx of a Toeplitz matrix T and a vector x, where
       T is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and
       its first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0].

    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix
        f_method (str): FFT method. 'numpy' for FFT from NumPy, 'recursive'
            for recursive method.

    Returns:
        np.ndarray: product of Toeplitz matrix and vector x

    """
    N = len(r)
    M = len(c)
    assert r[0] == c[0]
    assert len(x) == N
    n = int(2 ** np.ceil(np.log2(M + N - 1)))
    assert n >= M
    assert n >= N
    chat = np.r_[c, np.zeros(n - (M + N - 1)), r[-(N - 1):][::-1]]
    xhat = _zero_pad(x, n)
    yhat = _circulant_multiply(chat, xhat, f_method)
    y = yhat[:M]
    return y


def _toeplitz_mult_pd(r, c, x, f_method="numpy"):
    """Multiply Toeplitz matrix by vector using Pustylnikov's decomposition.

    See algorithm S3 in Sukhoy & Stoytchev 2019:

       Compute the product y = Tx of a Toeplitz matrix T and a vector x, where
       T is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and
       its first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0].

    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix
        f_method (str): FFT method. 'numpy' for FFT from NumPy, 'recursive'
            for recursive method.

    Returns:
        np.ndarray: product of Toeplitz matrix and vector x

    """
    N = len(r)
    M = len(c)
    assert r[0] == c[0]
    assert len(x) == N
    n = int(2 ** np.ceil(np.log2(M + N - 1)))
    if N != n:
        r = _zero_pad(r, n)
        x = _zero_pad(x, n)
    if M != n:
        c = _zero_pad(c, n)
    c1 = np.r_[c[0], c[1:]+r[-1:0:-1]]
    c2 = np.r_[c[0], c[1:]-r[-1:0:-1]]
    y1 = _circulant_multiply(c1 / 2, x, f_method)
    y2 = _skew_circulant_multiply(c2 / 2, x, f_method)
    y = y1[:M] + y2[:M]
    return y


def _zero_pad(x, n):
    """Zero pad an array x to length n by appending zeros.

    Args:
        x (np.ndarray): array x
        n (int): length of output array

    Returns:
        np.ndarray: array x with padding

    """
    m = len(x)
    assert m <= n
    xhat = np.zeros(n, dtype=complex)
    xhat[:m] = x
    return xhat


def _circulant_multiply(c, x, f_method="numpy"):
    """Multiply a circulant matrix by a vector.

    Runs in O(n log n) time.

    See algorithm S4 in Sukhoy & Stoytchev 2019:

       Compute the product y = Gx of a circulant matrix G and a vector x,
       where G is generated by its first column c=(c[0], c[1],...,c[n-1]).

    Args:
        c (np.ndarray): first column of circulant matrix G
        x (np.ndarray): vector x
        f_method (str): FFT method. 'numpy' for FFT from NumPy,
        'recursive' for recursive method.

    Returns:
        np.ndarray: product Gx

    """
    n = len(c)
    assert len(x) == n
    if f_method == "numpy":
        C = np.fft.fft(c)
        X = np.fft.fft(x)
        Y = C * X
        return np.fft.ifft(Y)
    elif f_method.lower() == "recursive":
        C = _fft(c)
        X = _fft(x)
        Y = C * X
        return _ifft(Y)
    else:
        print("f_method not recognized.")
        raise ValueError


def _skew_circulant_multiply(c, x, f_method="numpy"):
    """Multiply a skew-circulant matrix by a vector.

    Runs in O(n log n) time.

    See algorithm S7 in Sukhoy & Stoytchev 2019.

    Args:
        c (np.ndarray): first column of skew-circulant matrix G
        x (np.ndarray): vector x
        f_method (str): FFT method. 'numpy' for FFT from NumPy, 'recursive'
            for recursive method.

    Returns:
        np.ndarray: product Gx

    """
    n = len(c)
    assert len(x) == n
    k = np.arange(n, dtype=float)
    prefac = np.exp(-1j * k * np.pi / n)
    chat = c * prefac
    xhat = x * prefac
    y = _circulant_multiply(chat, xhat, f_method)
    y *= prefac.conjugate()
    return y


def _fft(x):
    """Recursive FFT algorithm. Runs in O(n log n) time.

    Args:
        x (np.ndarray): input

    Returns:
        np.ndarray: FFT of x

    """
    n = len(x)
    if n == 1:
        return x
    xe = x[0::2]
    xo = x[1::2]
    y1 = _fft(xe)
    y2 = _fft(xo)
    k = np.arange(n // 2)
    w = np.exp(-2j * np.pi * k / n)
    y = np.empty(n, dtype=complex)
    y[: n // 2] = y1 + w * y2
    y[n // 2:] = y1 - w * y2
    return y


def _ifft(y):
    """Recursive IFFT algorithm. Runs in O(n log n) time.

    Args:
        y (np.ndarray): input

    Returns:
        np.ndarray: IFFT of y

    """
    n = len(y)
    if n == 1:
        return y
    ye = y[0::2]
    yo = y[1::2]
    x1 = _ifft(ye)
    x2 = _ifft(yo)
    k = np.arange(n // 2)
    w = np.exp(2j * np.pi * k / n)
    x = np.empty(n, dtype=complex)
    x[: n // 2] = (x1 + w * x2) / 2
    x[n // 2:] = (x1 - w * x2) / 2
    return x


_available_t_methods = {
    "ce": _toeplitz_mult_ce,
    "pd": _toeplitz_mult_pd,
    "mm": lambda r, c, x, _: np.matmul(toeplitz(c, r), x),
    "scipy": lambda r, c, x, _: matmul_toeplitz((c, r), x),
}
