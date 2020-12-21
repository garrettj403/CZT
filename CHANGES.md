v0.0.2 (Dec 21, 2020)
=====================

- CZT:
	- Add function for inverse CZT (iczt) using complex conjugate.
	- ``czt``: 
		- Fixed default phase of W so that CZT provides FFT by default.
		- Wrapped ``czt_simple`` into ``czt`` (can still accessed via ``simple=True`` argument).
	- ``time2freq``:
		- Changed default frequency range to match FFT
		- Add normalization for arbitrary freq/time sweeps
	- Add functions to generate windows.
- Testing:
	- Add test to compare CZT to analytic expression
	- Added test to compare CZT to FFT
	- Added test for ICZT
	- Added test for time-domain -> frequency-domain -> time-domain conversion
	- Remove redundant tests
- Examples:
	- Add example calculating the time-domain response of a lossy waveguide
	- Automatic package reloading after each cell
- ``setup.py``:
	- Use ``py_modules`` instead of ``packages``

v0.0.1 (Dec 2, 2020)
====================

Initial release.
