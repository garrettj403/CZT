"""Chirp Z-transform.

Main reference:

    Sukhoy, V., Stoytchev, A. Generalizing the inverse FFT off the unit 
    circle. Sci Rep 9, 14443 (2019). 
    https://doi.org/10.1038/s41598-019-50234-9

"""

import numpy as np
from scipy.linalg import toeplitz


def czt(x, M=None, W=None, A=1.0, t_method='ce', f_method='std'):
    """Calculate Chirp Z-transform.

    Using an efficient algorithm. Solves in O(n log n) time.

    See algorithm 1 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        x (np.ndarray): input array
        M (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point
        t_method (str): Toeplitz matrix multiplication method. 'ce' for 
            circulant embedding, 'pd' for Pustylnikov's decomposition, 'mm'
            for simple matrix multiplication
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.

    Returns:
        np.ndarray: Chirp Z-transform

    """
    
    N = len(x)
    if M is None:
        M = N
    if W is None:
        W = np.exp(2j * np.pi / M)
        
    k = np.arange(N)
    X = W ** (k ** 2 / 2) * A ** -k * x
    r = W ** (-(k ** 2) / 2)
    k = np.arange(M)
    c = W ** (-(k ** 2) / 2)
    if t_method.lower() == 'ce':
        X = _toeplitz_mult_ce(r, c, X, f_method)
    elif t_method.lower() == 'pd':
        X = _toeplitz_mult_pd(r, c, X, f_method)
    elif t_method.lower() == 'mm':
        X = np.matmul(toeplitz(r, c), X)
    else:
        print("t_method not recognized.")
        raise ValueError
    for k in range(M):
        X[k] = W ** (k ** 2 / 2) * X[k]

    return X


# def iczt(X, N=None, W=None, A=1.0):

#     M = len(X)
#     if N is None:
#         N = M
#     if W is None:
#         W = np.exp(2j * np.pi / M)

#     assert M == N
#     n = N

#     # Multiply P^-1 and X
#     x = np.empty(n, dtype=complex)
#     for k in range(n):
#         x[k] = W ** (-(k ** 2) / 2) * X[k]

#     # Precompute the necessary polynomial products
#     p = np.empty(n, dtype=complex)
#     p[0] = 1
#     for k in range(1, n):
#         p[k] = p[k - 1] * (W ** k - 1)

#     # Compute the generating vector u
#     u = np.empty(n, dtype=complex)
#     for k in range(n):
#         u[k] = (-1)**k * W ** ((2*k**2 - (2*n-1)*k + n*(n-1)) / 2) / (p[n-k-1]*p[k])
    
#     z = np.zeros(n, dtype=complex)
#     uhat = np.r_[0, u[1:][::-1]]
#     util = np.r_[u[0], np.zeros(n - 1, dtype=complex)]
#     x1 = _toeplitz_mult_ce(uhat, z, x)
#     x1 = _toeplitz_mult_ce(z, uhat, x1)
#     x2 = _toeplitz_mult_ce(u, util, x)
#     x2 = _toeplitz_mult_ce(util, u, x2)

#     # Subtract and divide by u0
#     for k in range(n):
#         x[k] = (x2[k] - x1[k]) / u[0]

#     # multiply by A^-1 Q^-1
#     for k in range(n):
#         x[k] = A ** k * W * (-(k ** 2) / 2) * x[k]

#     return x


def czt_simple(x, M=None, W=None, A=1.0):
    """Simple Chirp Z-transform algorithm.

    Direct implementation. Brute force.

    Used to compare against czt in tests.

    Args:
        x (np.ndarray): input array
        M (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point

    Returns:
        np.ndarray: Chirp Z-transform

    """
    
    N = len(x)
    if M is None:
        M = N
    if W is None:
        W = np.exp(2j * np.pi / M)
    
    k = np.arange(M)
    X = np.zeros(M, dtype=complex)
    z = A * W ** -k
    for n in range(N):
        X += x[n] * z ** -n
    
    return X


# FREQ <--> TIME DOMAIN CONVERSION -------------------------------------------

def time_to_freq_domain(t, x, f=None):
    """Convert signal from time domain to frequency domain.

    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal

    Returns:
        np.ndarray: frequency-domain signal

    """

    # Input time
    t1, t2 = t.min(), t.max()  # start / stop time
    dt = t[1] - t[0]           # time step
    Nt = len(t)                # number of time points
    Fs = 1 / dt                # sampling frequency

    # Output frequency
    if f is None:
        f = np.linspace(0, Fs / 2, Nt)
    f1, f2 = f.min(), f.max()  # start / stop
    df = f[1] - f[0]           # frequency step
    bw = f2 - f1               # bandwidth
    Nf = len(f)                # number of frequency points

    # Step
    W = np.exp(-2j * np.pi * bw / Nf / Fs)

    # Starting point
    A = np.exp(2j * np.pi * f1 / Fs)

    # Frequency-domain transform
    freq_data = czt(x, Nf, W, A) #* dt

    return f, freq_data


def freq_to_time_domain(f, X, t=None):
    """Convert signal from frequency domain to time domain.

    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal

    Returns:
        np.ndarray: time-domain signal

    """

    # Input frequency
    f1, f2 = f.min(), f.max()  # start / stop frequency
    df = f[1] - f[0]           # frequency step
    bw = f2 - f1               # bandwidth
    fc = (f1 + f2) / 2         # center frequency
    Nf = len(f)                # number of frequency points
    t_alias = 1 / df           # alias-free interval

    # Output time
    if t is None:
        t = np.linspace(0, t_alias, Nf)
    t1, t2 = t.min(), t.max()  # start / stop time
    dt = t[1] - t[0]           # time step
    Nt = len(t)                # number of time points
    Fs = 1 / dt                # sampling frequency

    # Step
    W = np.exp(-2j * np.pi * bw / Nf / Fs)

    # Starting point
    A = np.exp(2j * np.pi * t1 / t_alias)

    # Time-domain transform
    time_data = np.conj(czt(np.conj(X), Nt, W, A))

    # Phase shift
    n = np.arange(Nt)
    phase = np.exp(2j * np.pi * f1 * n * dt)
    # phase = np.exp(2j * np.pi * (f1 + df / 2) * n * dt)

    return t, time_data * phase / Nf


# HELPER FUNCTIONS -----------------------------------------------------------

def _toeplitz_mult_ce(r, c, x, f_method='std'):
    """Multiply Toeplitz matrix by vector using circulant embedding.
    
    "Compute the product y = Tx of a Toeplitz matrix T and a vector x, where T
    is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and its 
    first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0]."
    
    See algorithm S1 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.
    
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
    chat = np.r_[c, np.zeros(n - (M + N - 1)), r[-(N-1):][::-1]]
    xhat = _zero_pad(x, n)
    yhat = _circulant_multiply(chat, xhat, f_method)
    y = yhat[:M]
    return y


def _toeplitz_mult_pd(r, c, x, f_method='std'):
    """Multiply Toeplitz matrix by vector using Pustylnikov's decomposition.
    
    Compute the product y = Tx of a Toeplitz matrix T and a vector x, where T
    is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and its 
    first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0].
    
    See algorithm S3 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.
    
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
    c1 = np.empty(n, dtype=complex)
    c2 = np.empty(n, dtype=complex)
    c1[0] = 0.5 * c[0]
    c2[0] = 0.5 * c[0]
    for k in range(1, n):
        c1[k] = 0.5 * (c[k] + r[n - k])
        c2[k] = 0.5 * (c[k] - r[n - k])
    y1 = _circulant_multiply(c1, x, f_method)
    y2 = _skew_circulant_multiply(c2, x, f_method)
    y = y1[:M] + y2[:M]
    return y


def _zero_pad(x, n):
    """Zero pad an array x to length n by appending zeros.
    
    See algorithm S2 in Sukhoy & Stoytchev 2019 (full reference in README).
    
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


def _circulant_multiply(c, x, f_method='std'):
    """Multiply a circulat matrix by a vector.
    
    Compute the product y = Gx of a circulat matrix G and a vector x, where G 
    is generated by its first column c=(c[0], c[1],...,c[n-1]).

    Runs in O(n log n) time.

    See algorithm S4 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        c (np.ndarray): first column of circulant matrix G
        x (np.ndarray): vector x
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.

    Returns:
        np.ndarray: product Gx
    
    """
    n = len(c)
    assert len(x) == n
    if f_method == 'std':
        C = np.fft.fft(c)
        X = np.fft.fft(x)
        Y = np.empty(n, dtype=complex)
        for k in range(n):
            Y[k] = C[k] * X[k]
        y = np.fft.ifft(Y)
    elif f_method.lower() == 'fast':
        C = _fft(c)
        X = _fft(x)
        Y = np.empty(n, dtype=complex)
        for k in range(n):
            Y[k] = C[k] * X[k]
        y = _ifft(Y)
    else:
        print("f_method not recognized.")
        raise ValueError
    return y


def _skew_circulant_multiply(c, x, f_method='std'):
    """Multiply a skew-circulant matrix by a vector.
    
    Runs in O(n log n) time.
    
    See algorithm S7 in Sukhoy & Stoytchev 2019 (full reference in README).
    
    Args:
        c (np.ndarray): first column of skew-circulant matrix G
        x (np.ndarray): vector x
        f_method (str): FFT method. 'std' for standard FFT (from NumPy), or 
            'fast' for a method that may be faster for large arrays.

    Returns:
        np.ndarray: product Gx

    """
    n = len(c)
    assert len(x) == n
    chat = np.empty(n, dtype=complex)
    xhat = np.empty(n, dtype=complex)
    for k in range(n):
        chat[k] = c[k] * np.exp(-1j * k * np.pi / n)
        xhat[k] = x[k] * np.exp(-1j * k * np.pi / n)
    # k = np.arange(n, dtype=complex)
    # chat = c * np.exp(-1j * k * np.pi / n)
    # xhat = c * np.exp(-1j * k * np.pi / n)
    y = _circulant_multiply(chat, xhat, f_method)
    for k in range(n):
        y[k] = y[k] * np.exp(1j * k * np.pi / n)
    return y


def _fft(x):
    """FFT algorithm. Runs in O(n log n) time.

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
    y1 = np.fft.fft(xe)
    y2 = np.fft.fft(xo)
    # Todo: simplify
    y = np.zeros(n, dtype=complex)
    for k in range(n // 2 - 1):
        w = np.exp(-2j * np.pi * k / n)
        y[k]            = y1[k] + w * y2[k]
        y[k + (n // 2)] = y1[k] - w * y2[k]
    return y


def _ifft(y):
    """IFFT algorithm. Runs in O(n log n) time.

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
    x1 = np.fft.ifft(ye)
    x2 = np.fft.ifft(yo)
    # TODO: simplify
    x = np.zeros(n, dtype=complex)
    for k in range(n // 2 -1):
        w = np.exp(2j * np.pi * k / n)
        x[k]            = (x1[k] + w * x2[k]) / 2
        x[k + (n // 2)] = (x1[k] - w * x2[k]) / 2
    return x
