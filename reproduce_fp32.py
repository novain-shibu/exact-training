"""Same test as reproduce.py, but every exact matmul is computed with FLOAT32 ONLY - so it can run on GPUs that have no float64 or
int8 matmul (Apple GPUs via MPS, many mobile and consumer GPUs). Usage:  python reproduce_fp32.py [numpy|mps|cuda|cpu]
Method: each integer operand (|x| <= 2^15) is split into two signed base-128 digits; digit products are < 2^16, and at most 128 of them
are summed per float32 matmul, so every partial sum is an integer below 2^23 and float32 returns it exactly in any order. This slicing
is the idea of the Ozaki scheme (exact pieces on low-precision hardware). The exact hash must equal reproduce.py's: ced279b58bc72962."""
import sys, numpy as np
import xtrain as X
import reproduce as R

DEV = sys.argv[1] if len(sys.argv) > 1 else "numpy"
KC = 128                                               # 256 * 256 * 128 = 2^23 < 2^24

def split7(x):
    lo = ((x + 64) & 127) - 64
    return (x - lo) >> 7, lo

if DEV == "numpy":
    def mm32(x, y): return np.asarray(x, np.float32) @ np.asarray(y, np.float32)
else:
    import torch
    dev = torch.device(DEV)
    if DEV == "cuda": torch.backends.cuda.matmul.allow_tf32 = False      # digits are exact even in TF32, but keep plain fp32
    def mm32(x, y):
        return (torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev) @ torch.from_numpy(np.ascontiguousarray(y, np.float32)).to(dev)).cpu().numpy()

def exact_product_fp32(aq, bq):
    assert np.abs(aq).max() <= 1 << 15 and np.abs(bq).max() <= 1 << 15
    ah, al = split7(aq); bh, bl = split7(bq); K = aq.shape[1]; out = np.zeros((aq.shape[0], bq.shape[1]), np.int64)
    for k0 in range(0, K, KC):
        s = slice(k0, k0 + KC)
        c = lambda x, y: np.rint(mm32(x, y)).astype(np.int64)
        out += (c(ah[:, s], bh[s]) << 14) + ((c(ah[:, s], bl[s]) + c(al[:, s], bh[s])) << 7) + c(al[:, s], bl[s])
    return out

_orig_mm = X.Arith.mm
def mm(s, a, b, fa, fb, fo):
    if not s.int: return _orig_mm(s, a, b, fa, fb, fo)
    aq, sa = X.q_axis(a, 1); bq, sb = X.q_axis(b, 0)
    return X.shr(exact_product_fp32(aq, bq), (fa + fb - fo) - sa - sb)
X.Arith.mm = mm

if __name__ == "__main__":
    d = R.corpus()
    e1 = R.run("int", d, False)
    print(f"device: {DEV}\nexact (float32-only matmuls) : {e1}\nexact hash vs reproduce.py ({R.EXPECTED}): {'MATCH' if e1 == R.EXPECTED else 'NO MATCH'}")
