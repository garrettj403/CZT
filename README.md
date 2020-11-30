Chirp z-Transform (CZT)
=======================

Example
-------

Consider the following time-domain signal:

<p align="center">
<img src="https://raw.githubusercontent.com/garrettj403/CZT/main/examples/results/signal.png" width="500">
</p>

This is an exponentially decaying sine wave with some distortion from higher-order frequencies. We can convert the signal to the frequency-domain to investigate the frequency content using the Inverse Chirp z-Transform (ICZT):

<p align="center">
<img src="https://raw.githubusercontent.com/garrettj403/CZT/main/examples/results/freq-domain.png" width="500">
</p>

Note that the ICZT also allows us to calculate the frequency response over an arbitrary frequency range:

<p align="center">
<img src="https://raw.githubusercontent.com/garrettj403/CZT/main/examples/results/zoom-czt.png" width="500">
</p>

We can see that the signal has frequency components at 1 kHz, 2.5 kHz and 3.5 kHz. To remove the distortion and isolate the 1 kHz signal, we can apply a simple window in the frequency-domain:

<p align="center">
<img src="https://raw.githubusercontent.com/garrettj403/CZT/main/examples/results/windowed-freq-domain.png" width="500">
</p>

Finally, we can use the CZT to trasform back to the time domain:

<p align="center">
<img src="https://raw.githubusercontent.com/garrettj403/CZT/main/examples/results/windowed-time-domain.png" width="500">
</p>

As we can see, we were able to remove the higher-order frequencies that were distorting our 1 kHz signal.

References
----------

- [Rabiner, L., Schafer, R., Rader, C. The Chirp z-Transform Algorithm. IEEE Trans. Audio Electroacoustics, Au-17, 2, Jun. 1969.](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/015_czt.pdf)

- Sukhoy, V., Stoytchev, A. Generalizing the inverse FFT off the unit circle. Sci Rep 9, 14443 (2019). https://doi.org/10.1038/s41598-019-50234-9

- [Chirp Z-Transform (Wikipedia)](https://en.wikipedia.org/wiki/Chirp_Z-transform)

- [Discrete Fourier Transform (Wikipedia)](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
