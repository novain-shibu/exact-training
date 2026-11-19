# Exact training: same model, same bits on every machine we tested

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24370497.svg)](https://doi.org/10.5281/zenodo.24370497)

`python reproduce.py` (Python 3 + numpy, CPU only, about 3 minutes)

It trains one tiny transformer four times on the same deterministic data:
- twice in ordinary float32, changing only the order in which the batch rows are summed;
- twice in exact integer arithmetic (xtrain.py), with the same change.

Expected output: the two float32 hashes differ; the two exact hashes are identical and equal `ced279b58bc72962`.

Checked so far: a Windows laptop (Intel Core Ultra, Python 3.12, numpy 2.5.3) and a Linux cloud CPU (Intel Xeon @ 2.20GHz,
Python 3.13, numpy 2.1.3, default and 1 BLAS thread). If your exact hash is different, please open an issue with your CPU, OS,
Python and numpy versions - that would be a bug in our claim. Reports from ARM, Apple, AMD and other CPUs are especially welcome.

Nothing in the exact path depends on the machine's maths library: training is int64 arithmetic (matmul products are exact
integers below 2^53, so float64 BLAS returns them exactly in any order); the softmax exp table is computed with Python's decimal module
at 50 digits, rounded to integers and pinned by a checksum (no libm); data and starting weights come from splitmix64 integers, with a few
IEEE-754 basic operations per weight (division, multiplication, square root - all correctly rounded by the standard). No numpy random generator is used.

Results and limits: see [NOTE.md](NOTE.md).

Files: `NOTE.md` (results, limits, related work), `reproduce.py` (the test), `xtrain.py` (the exact integer trainer: forward, backward, SGD-momentum, Lion, block-float Adam).
License: Apache-2.0, same as exact-llm. The author is a student; replies to issues may take about a week.
Cite: Shibu, N. (2026). *Same bits on every chip we tested: exact integer training, unlearning and RL* (v0.1.0). Zenodo.
https://doi.org/10.5281/zenodo.24370497 (all versions: https://doi.org/10.5281/zenodo.24370496). See also `CITATION.cff`. Related exact-inference work by the same author: exact-llm (doi.org/10.5281/zenodo.24137507).
