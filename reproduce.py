"""Reproduce it yourself (CPU, numpy only, a few minutes).
Trains the same tiny transformer twice in float32 (identical data and init; only the order in which batch rows are summed differs) and
twice with exact integer arithmetic. Float runs end with different weights; exact runs end with identical weights - and the exact hash
must equal EXPECTED below on ANY machine (CPU vendor, OS, BLAS, thread count, numpy version). Nothing in the exact
path uses libm or numpy's random generators: data, initial weights and the exp table are built from integers or correctly rounded ops. Usage:  python reproduce.py"""
import hashlib, math, numpy as np, xtrain as X

EXPECTED = "ced279b58bc72962"                     # exact weight hash from the author's run; yours must match
MASK = (1 << 64) - 1
def splitmix(i, seed=12345):
    z = (seed + (i + 1) * 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK; z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)
def corpus(n=200_000):
    """Deterministic synthetic text: a fixed 2nd-order byte Markov chain (same bytes on every machine)."""
    alpha = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz .,", np.uint8); k = len(alpha); out = np.zeros(n, np.int64); a = b = 0
    for i in range(n):
        r = splitmix(i); c = (a * 7 + b * 3 + (r % 5 == 0) * (r >> 8)) % k; out[i] = alpha[c]; a, b = b, c
    return out
def gauss(n, stream):
    """Portable N(0,1)-like draws: sum of 4 uniform 24-bit integers from splitmix64 (Irwin-Hall), integer-centred, then one
    division and one multiply in IEEE float64 (correctly rounded everywhere). No numpy RNG, no libm: same values on every machine."""
    i = np.arange(4 * n, dtype=np.uint64) + np.uint64(stream << 32); M = np.uint64
    with np.errstate(over="ignore"):
        z = M(12345) + (i + M(1)) * M(0x9E3779B97F4A7C15)
        z = (z ^ (z >> M(30))) * M(0xBF58476D1CE4E5B9); z = (z ^ (z >> M(27))) * M(0x94D049BB133111EB); z = z ^ (z >> M(31))
    u = (z >> M(40)).astype(np.int64).reshape(n, 4).sum(1) - (1 << 25)          # exact integer, mean 0, sd 2^24/sqrt(3)
    return u.astype(np.float64) / float(1 << 24) * 1.7320508075688772
def init_params(cfg):
    """Same shapes and scales as xtrain.init_params, drawn with gauss() so the starting weights are bit-identical on every machine."""
    d, ff, L, V, T = cfg["d"], cfg["ff"], cfg["L"], cfg["V"], cfg["T"]; k = [0]
    def g(shape, scale):
        k[0] += 1; return gauss(int(np.prod(shape)), k[0]).reshape(shape) * scale
    p = {"E": g((V, d), 0.1), "P": g((T, d), 0.02), "Wout": g((d, V), 1 / math.sqrt(d) * 0.5)}
    for l in range(L):
        for n in ("Wq", "Wk", "Wv"): p[f"{n}{l}"] = g((d, d), 1 / math.sqrt(d))
        p[f"Wo{l}"] = g((d, d), 1 / math.sqrt(d) / math.sqrt(2 * L))
        p[f"W1{l}"] = g((d, ff), 1 / math.sqrt(d))
        p[f"W2{l}"] = g((ff, d), 1 / math.sqrt(ff) / math.sqrt(2 * L))
    return p
def run(mode, data, reorder, steps=400):
    cfg = dict(d=64, H=4, ff=256, L=2, V=256, T=32); B = 8; m = X.Model(cfg, mode, init_params(cfg))
    for s in range(steps):
        st = np.array([splitmix(10_000_000 + s * B + j) % (len(data) - cfg["T"] - 1) for j in range(B)], np.int64)
        batch = data[st[:, None] + np.arange(cfg["T"] + 1)]
        if reorder: batch = batch[[(j * 5 + s) % B for j in range(B)]]            # same rows, different summation order
        _, g = m.loss_and_grad(batch[:, :-1], batch[:, 1:]); m.sgd(g, 5, 8)
    return m.weights_hash()
if __name__ == "__main__":
    d = corpus()
    f1, f2 = run("f32", d, False), run("f32", d, True)
    e1, e2 = run("int", d, False), run("int", d, True)
    print(f"float32 run A : {f1}\nfloat32 run B : {f2}   -> {'IDENTICAL' if f1 == f2 else 'DIFFERENT (same data, only the summation order changed)'}")
    print(f"exact   run A : {e1}\nexact   run B : {e2}   -> {'IDENTICAL' if e1 == e2 else 'DIFFERENT'}")
    print(f"exact hash vs the author's machine ({EXPECTED}): {'MATCH' if e1 == EXPECTED else 'NO MATCH'}")
