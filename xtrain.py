# xtrain: one tiny decoder-only transformer, trained either in float (float32/float64) or as an EXACT INTEGER program.
# The same code path computes forward, backward and SGD-momentum for both modes; only the arithmetic primitives differ.
# Integer mode: activations / activation-gradients Q24, weights / weight-gradients / momentum Q32, all int64.
# Every matmul quantises its operands to 15-bit mantissas with power-of-two shifts per row (left) / per column (right) - block floating
# point - and multiplies them EXACTLY (products < 2^53, so a float64 BLAS matmul returns exact integers in any summation order).
# Softmax uses an integer exp table; rounding is round-half-up shifts. Every integer sum is exact => results are independent of
# summation order, thread count and hardware. Architecture (both modes): token + position embedding, L x [causal multi-head softmax
# attention, ReLU MLP], no normalisation layers, untied output head.
import numpy as np, math, hashlib

FA, FW = 24, 32                                     # fixed-point fractional bits: activations, weights
MANT = 15
MM_BACKEND = ["cpu"]
EXP_STEP = 8
def _exp_table():
    """exp(-i/256) * 2^24, correctly rounded via Python decimal (no libm, so identical on every platform); pinned by a checksum."""
    from decimal import Decimal, localcontext, ROUND_HALF_EVEN
    with localcontext() as c:
        c.prec = 50; t = [int(((Decimal(-i) / 256).exp() * (1 << 24)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)) for i in range(256 * 40 + 2)]
    assert hashlib.sha256(",".join(map(str, t)).encode()).hexdigest()[:16] == "159660e3e43eda62", "exp table checksum mismatch"
    return np.array(t, np.int64)
EXP_T = _exp_table()

def shr(a, n):
    """round-half-up arithmetic shift right by n (array or int; negative n = left shift)."""
    n = np.asarray(n, dtype=np.int64)
    if n.ndim == 0:
        n = int(n)
        return a << (-n) if n <= 0 else (a + (1 << (n - 1))) >> n
    pos = np.maximum(n, 0); neg = np.maximum(-n, 0)
    r = np.where(pos > 0, (a + (np.int64(1) << np.maximum(pos - 1, 0))) >> pos, a)
    return r << neg

def bitlen(m):
    m = np.asarray(m, dtype=np.int64); out = np.zeros_like(m); v = m.copy()
    for b in (32, 16, 8, 4, 2, 1):
        big = v >= (np.int64(1) << b); out += big * b; v = np.where(big, v >> b, v)
    return out + (v > 0)

def q_axis(x, axis):
    """quantise to MANT-bit mantissa with a power-of-two shift per row (axis=1) or per column (axis=0)."""
    m = np.abs(x).max(axis=axis, keepdims=True)
    s = np.maximum(bitlen(m) - MANT, 0)
    return shr(x, s), s

class Arith:
    def __init__(s, mode):
        s.mode = mode; s.int = mode == "int"; s.dt = {"f32": np.float32, "f64": np.float64}.get(mode)
    # matmul with explicit input/output scales (ignored in float modes)
    def mm(s, a, b, fa, fb, fo):
        if not s.int: return a @ b
        aq, sa = q_axis(a, 1); bq, sb = q_axis(b, 0)
        assert a.shape[1] <= 4096
        if MM_BACKEND[0] == "cuda":                                   # same exact product on a GPU (float64 cuBLAS; exact below 2^53)
            import torch
            p = torch.round(torch.from_numpy(aq).cuda().double() @ torch.from_numpy(bq).cuda().double()).long().cpu().numpy()
        else:
            p = np.rint(aq.astype(np.float64) @ bq.astype(np.float64)).astype(np.int64)  # exact: |p| < 2^30 * 4096 < 2^53
        return shr(p, (fa + fb - fo) - sa - sb)
    def mul(s, a, b, fa, fb, fo):          # elementwise product
        return a * b if not s.int else shr(a * b, fa + fb - fo)
    def softmax(s, x, mask):               # x activations (FA), mask bool; returns probabilities (FA)
        if not s.int:
            x = np.where(mask, x, -np.inf); x = x - x.max(-1, keepdims=True); e = np.exp(x); return (e / e.sum(-1, keepdims=True)).astype(s.dt)
        x16 = shr(x, FA - 16)
        mx = np.where(mask, x16, np.iinfo(np.int64).min // 4).max(-1, keepdims=True)
        d = np.where(mask, mx - x16, 0)
        i = np.minimum(d >> EXP_STEP, len(EXP_T) - 2); fr = d & ((1 << EXP_STEP) - 1)
        e = EXP_T[i] - shr((EXP_T[i] - EXP_T[i + 1]) * fr, EXP_STEP)
        e = np.where(mask & (d < (40 << 16)), e, 0)
        return (e << FA) // e.sum(-1, keepdims=True)
    def from_float(s, x, f):
        return np.rint(np.asarray(x, np.float64) * (1 << f)).astype(np.int64) if s.int else np.asarray(x, s.dt)
    def to_float(s, x, f):
        return x.astype(np.float64) / (1 << f) if s.int else x.astype(np.float64)

def init_params(cfg, seed=0):
    r = np.random.default_rng(seed); d, ff, L, V, T = cfg["d"], cfg["ff"], cfg["L"], cfg["V"], cfg["T"]
    p = {"E": r.standard_normal((V, d)) * 0.1, "P": r.standard_normal((T, d)) * 0.02, "Wout": r.standard_normal((d, V)) / math.sqrt(d) * 0.5}
    for l in range(L):
        for n in ("Wq", "Wk", "Wv"): p[f"{n}{l}"] = r.standard_normal((d, d)) / math.sqrt(d)
        p[f"Wo{l}"] = r.standard_normal((d, d)) / math.sqrt(d) / math.sqrt(2 * L)
        p[f"W1{l}"] = r.standard_normal((d, ff)) / math.sqrt(d)
        p[f"W2{l}"] = r.standard_normal((ff, d)) / math.sqrt(ff) / math.sqrt(2 * L)
    return p

class Model:
    def __init__(s, cfg, mode, p_float):
        s.c = cfg; s.A = Arith(mode); s.p = {k: s.A.from_float(v, FW) for k, v in p_float.items()}
        s.m = {k: np.zeros_like(v) for k, v in s.p.items()}
        hd = cfg["d"] // cfg["H"]; s.scale_f = 1 / math.sqrt(hd); s.SC = int(round((1 << 16) / math.sqrt(hd)))   # integer attention scale, Q16
    def W(s, k): return s.p[k]
    def scale(s, x):                          # multiply attention scores by 1/sqrt(hd)
        return x * s.scale_f if not s.A.int else shr(x * s.SC, 16)
    def forward(s, tok):
        A, c = s.A, s.c; B, T = tok.shape; d, H = c["d"], c["H"]; hd = d // H; cache = {}
        x = (s.p["E"][tok] + s.p["P"][None, :T]).reshape(B * T, d)
        if A.int: x = shr(x, FW - FA)
        mask = np.tril(np.ones((T, T), bool))
        for l in range(c["L"]):
            q = A.mm(x, s.W(f"Wq{l}"), FA, FW, FA); k = A.mm(x, s.W(f"Wk{l}"), FA, FW, FA); v = A.mm(x, s.W(f"Wv{l}"), FA, FW, FA)
            qh = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3); kh = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3); vh = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            S = np.stack([np.stack([s.scale(A.mm(qh[b, h], kh[b, h].T, FA, FA, FA)) for h in range(H)]) for b in range(B)])
            P_ = A.softmax(S, mask)
            O = np.stack([np.stack([A.mm(P_[b, h], vh[b, h], FA, FA, FA) for h in range(H)]) for b in range(B)])
            Oc = O.transpose(0, 2, 1, 3).reshape(B * T, d)
            x1 = x + A.mm(Oc, s.W(f"Wo{l}"), FA, FW, FA)
            a = A.mm(x1, s.W(f"W1{l}"), FA, FW, FA); h_ = np.maximum(a, 0)
            x2 = x1 + A.mm(h_, s.W(f"W2{l}"), FA, FW, FA)
            cache[l] = (x, qh, kh, vh, P_, Oc, x1, a, h_); x = x2
        logits = A.mm(x, s.W("Wout"), FA, FW, FA)
        cache["xL"] = x; cache["mask"] = mask
        return logits, cache
    def loss_and_grad(s, tok, tgt):
        A, c = s.A, s.c; B, T = tok.shape; d, H = c["d"], c["H"]; hd = d // H; V = c["V"]
        logits, cache = s.forward(tok)
        lf = A.to_float(logits, FA); lf = lf - lf.max(-1, keepdims=True)
        loss = float(np.mean(np.log(np.exp(lf).sum(-1)) - lf[np.arange(B * T), tgt.reshape(-1)]))
        allmask = np.ones((B * T, V), bool)
        prob = A.softmax(logits, allmask)
        onehot = np.zeros_like(prob); onehot[np.arange(B * T), tgt.reshape(-1)] = (1 << FA) if A.int else 1
        N = B * T
        dlog = prob - onehot                                                   # gradient of SUM of token losses
        g = {}
        def wgrad(xin, dout):                                                  # x^T @ dout, scaled by 1/N (mean loss), weight-grad scale FW
            if not A.int: return (xin.T @ dout) / N
            gw = A.mm(xin.T, dout, FA, FA, FW + 0)
            return gw // N if False else _div_round(gw, N)
        g["Wout"] = wgrad(cache["xL"], dlog)
        dx = A.mm(dlog, s.W("Wout").T, FA, FW, FA)
        for l in reversed(range(c["L"])):
            x, qh, kh, vh, P_, Oc, x1, a, h_ = cache[l]
            g[f"W2{l}"] = wgrad(h_, dx)
            dh = A.mm(dx, s.W(f"W2{l}").T, FA, FW, FA); da = np.where(a > 0, dh, 0)
            g[f"W1{l}"] = wgrad(x1, da)
            dx1 = dx + A.mm(da, s.W(f"W1{l}").T, FA, FW, FA)
            g[f"Wo{l}"] = wgrad(Oc, dx1)
            dOc = A.mm(dx1, s.W(f"Wo{l}").T, FA, FW, FA)
            dO = dOc.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            dq = np.zeros_like(qh); dk = np.zeros_like(kh); dv = np.zeros_like(vh)
            for b in range(B):
                for hh in range(H):
                    P = P_[b, hh]
                    dP = A.mm(dO[b, hh], vh[b, hh].T, FA, FA, FA)
                    dv[b, hh] = A.mm(P.T, dO[b, hh], FA, FA, FA)
                    rs = A.mul(P, dP, FA, FA, FA).sum(-1, keepdims=True)
                    dS = s.scale(A.mul(P, dP - rs, FA, FA, FA))
                    dq[b, hh] = A.mm(dS, kh[b, hh], FA, FA, FA)
                    dk[b, hh] = A.mm(dS.T, qh[b, hh], FA, FA, FA)
            dqf = dq.transpose(0, 2, 1, 3).reshape(B * T, d); dkf = dk.transpose(0, 2, 1, 3).reshape(B * T, d); dvf = dv.transpose(0, 2, 1, 3).reshape(B * T, d)
            g[f"Wq{l}"] = wgrad(x, dqf); g[f"Wk{l}"] = wgrad(x, dkf); g[f"Wv{l}"] = wgrad(x, dvf)
            dx = dx1 + A.mm(dqf, s.W(f"Wq{l}").T, FA, FW, FA) + A.mm(dkf, s.W(f"Wk{l}").T, FA, FW, FA) + A.mm(dvf, s.W(f"Wv{l}").T, FA, FW, FA)
        dx0 = dx.reshape(B, T, d)
        if A.int:
            dx0w = shr(dx0, FA - FW)                                           # to weight scale (left shift: FW > FA)
            gE = np.zeros_like(s.p["E"]); np.add.at(gE, tok.reshape(-1), dx0w.reshape(B * T, d)); g["E"] = _div_round(gE, N)
            g["P"] = _div_round(np.zeros_like(s.p["P"]) + np.pad(dx0w.sum(0), ((0, c["T"] - T), (0, 0))), N)
        else:
            gE = np.zeros_like(s.p["E"]); np.add.at(gE, tok.reshape(-1), dx0.reshape(B * T, d)); g["E"] = gE / N
            g["P"] = np.pad(dx0.sum(0), ((0, c["T"] - T), (0, 0))) / N
        return loss, g
    def sgd(s, g, lr_num, lr_shift, beta_num=29491):
        """momentum SGD: m = beta*m + g ; W -= lr*m ; beta = beta_num/2^15, lr = lr_num/2^lr_shift (identical values in both modes)."""
        for k in s.p:
            if s.A.int:
                s.m[k] = shr(s.m[k] * beta_num, 15) + g[k]
                s.p[k] = s.p[k] - shr(s.m[k] * lr_num, lr_shift)
            else:
                s.m[k] = s.m[k] * s.A.dt(beta_num / 32768) + g[k].astype(s.A.dt)
                s.p[k] = s.p[k] - s.A.dt(lr_num / 2 ** lr_shift) * s.m[k]
    def adam(s, g, lr_num, lr_shift, b1_num=29491, b2_num=32735, eps_q24=168):
        """Adam without bias correction (identical formula in both modes): m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2 ;
        W -= lr * m / (sqrt(v) + eps).  b1 = b1_num/2^15, b2 = b2_num/2^15, eps = eps_q24/2^24 (~1e-5), lr = lr_num/2^lr_shift.
        Integer mode: m Q32, v Q48 (g^2 >> 16), sqrt(v) Q24 via exact integer square root, ratio Q24 by floor division."""
        if not hasattr(s, "v"): s.v = {k: np.zeros_like(v) for k, v in s.p.items()}
        for k in s.p:
            if s.A.int:
                s.m[k] = s.m[k] - shr(s.m[k] * (32768 - b1_num), 15) + shr(g[k] * (32768 - b1_num), 15)   # decay via the small complement: no overflow
                g8 = shr(g[k], 8); s.v[k] = s.v[k] - shr(s.v[k] * (32768 - b2_num), 15) + shr((g8 * g8) * (32768 - b2_num), 15)   # (g>>8)^2 = Q48, no int64 overflow for |grad| < 2^23
                assert int(s.v[k].min()) >= 0, "Adam v overflow"
                rt = _isqrt(s.v[k])                                        # sqrt of Q48 -> Q24
                ratio = np.floor_divide(s.m[k] << 16, rt + eps_q24)        # Q32+16-24 = Q24
                s.p[k] = s.p[k] - shr(ratio * lr_num, lr_shift - 8)       # Q24 -> Q32 with lr
            else:
                dt = s.A.dt; gk = g[k].astype(dt)
                s.m[k] = s.m[k] * dt(b1_num / 32768) + dt(1 - b1_num / 32768) * gk
                s.v[k] = s.v[k] * dt(b2_num / 32768) + dt(1 - b2_num / 32768) * gk * gk
                s.p[k] = s.p[k] - dt(lr_num / 2 ** lr_shift) * s.m[k] / (np.sqrt(s.v[k]) + dt(eps_q24 / 2 ** 24))
    def adam_bf(s, g, lr_num, lr_shift, b1_num=29491, b2_num=32735, eps_q24=168):
        """Adam round 2 (2026-11-14): same formula as adam(), but the second moment v is kept in per-tensor BLOCK FLOATING POINT
        (int64 mantissa < 2^46 times 2^ev, in units of Q64 g^2) so small gradients never underflow to v = 0.
        Float modes: identical to adam()."""
        if not s.A.int: return s.adam(g, lr_num, lr_shift, b1_num, b2_num, eps_q24)
        if not hasattr(s, "vm"): s.vm = {k: np.zeros_like(v) for k, v in s.p.items()}; s.ev = {k: 0 for k in s.p}
        EPS32 = eps_q24 << 8                                               # eps in Q32
        for k in s.p:
            gk = g[k]
            s.m[k] = s.m[k] - shr(s.m[k] * (32768 - b1_num), 15) + shr(gk * (32768 - b1_num), 15)
            sg = max(int(bitlen(np.array([np.abs(gk).max()]))[0]) - 30, 0); gs = shr(gk, sg); g2 = gs * gs; e2 = 2 * sg   # g^2 = g2 * 2^e2 (Q64)
            top = max(int(bitlen(np.array([s.vm[k].max()]))[0]) + s.ev[k], int(bitlen(np.array([g2.max()]))[0]) + e2)
            e = top - 46
            a = shr(s.vm[k], e - s.ev[k]); b = shr(g2, e - e2)
            s.vm[k] = a - shr(a * (32768 - b2_num), 15) + shr(b * (32768 - b2_num), 15); s.ev[k] = e
            kk = 14 if (e - 14) % 2 == 0 else 15
            rt = _isqrt(s.vm[k] << kk); er = (e - kk) // 2               # sqrt(v) in Q32 = rt * 2^er
            den = rt + shr(np.full_like(rt, EPS32), er) if er > -40 else np.full_like(rt, 1 << 62)
            t = 24 - er                                                     # ratio Q24 = m * 2^(24-er) / den
            assert int(np.abs(s.m[k]).max()) < (1 << 32), "Adam m range"
            if t > 30: ratio = np.floor_divide(s.m[k] << 30, np.maximum(shr(den, t - 30), 1))
            elif t >= 0: ratio = np.floor_divide(s.m[k] << t, den)
            else: ratio = np.floor_divide(shr(s.m[k], -t), den)
            s.p[k] = s.p[k] - shr(ratio * lr_num, lr_shift - 8)
    def adam_ieee(s, g, lr_num, lr_shift, b1_num=29491, b2_num=32735, eps_q24=168):
        """Unconventional variant: exact integer gradients, optimizer arithmetic in IEEE-754 float64 ELEMENTWISE ops only (+ - * / sqrt are
        correctly rounded, so with no reductions and no fused ops the result is bit-identical on any IEEE platform). Weights stay Q32 int."""
        if not s.A.int: return s.adam(g, lr_num, lr_shift, b1_num, b2_num, eps_q24)
        if not hasattr(s, "mf"): s.mf = {k: np.zeros(v.shape) for k, v in s.p.items()}; s.vf = {k: np.zeros(v.shape) for k, v in s.p.items()}
        b1 = b1_num / 32768; b2 = b2_num / 32768; eps = eps_q24 / 2 ** 24; lr = lr_num / 2 ** lr_shift
        for k in s.p:
            gk = g[k].astype(np.float64) / 4294967296.0                       # exact (|g| < 2^53)
            s.mf[k] = s.mf[k] * b1 + gk * (1 - b1)
            s.vf[k] = s.vf[k] * b2 + (gk * gk) * (1 - b2)
            upd = (s.mf[k] / (np.sqrt(s.vf[k]) + eps)) * lr
            s.p[k] = s.p[k] - np.rint(upd * 4294967296.0).astype(np.int64)
    def lion(s, g, lr_num, lr_shift, b1_num=29491, b2_num=32440):
        """Lion (Chen et al. 2023): W -= lr * sign(b1 m + (1-b1) g); m = b2 m + (1-b2) g. Needs only signs - natively integer.
        b1 = 0.9, b2 = 0.99 (b2_num/2^15). Identical formula in all modes."""
        for k in s.p:
            if s.A.int:
                c = s.m[k] - shr(s.m[k] * (32768 - b1_num), 15) + shr(g[k] * (32768 - b1_num), 15)
                s.p[k] = s.p[k] - np.sign(c) * shr(np.int64(1 << 32) * lr_num, lr_shift)
                s.m[k] = s.m[k] - shr(s.m[k] * (32768 - b2_num), 15) + shr(g[k] * (32768 - b2_num), 15)
            else:
                dt = s.A.dt; gk = g[k].astype(dt)
                c = s.m[k] * dt(b1_num / 32768) + dt(1 - b1_num / 32768) * gk
                s.p[k] = s.p[k] - dt(lr_num / 2 ** lr_shift) * np.sign(c)
                s.m[k] = s.m[k] * dt(b2_num / 32768) + dt(1 - b2_num / 32768) * gk
    def opt(s, name, g, lr_num, lr_shift):
        return getattr(s, name)(g, lr_num, lr_shift)
    def weights_hash(s):
        h = hashlib.sha256()
        for k in sorted(s.p): h.update(np.ascontiguousarray(s.p[k]).tobytes())
        return h.hexdigest()[:16]

def _div_round(x, n):
    """exact integer division by n with round-half-up (deterministic)."""
    return np.floor_divide(x * 2 + n, 2 * n)

def _isqrt(n):
    """exact floor square root of non-negative int64 array (< 2^62)."""
    r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
    r = np.where(r * r > n, r - 1, r); r = np.where(r * r > n, r - 1, r)
    r = np.where((r + 1) * (r + 1) <= n, r + 1, r); return np.where((r + 1) * (r + 1) <= n, r + 1, r)
