# Same bits on every chip we tested: exact integer training, unlearning and RL

**Novain Shibu** — github.com/novain-shibu/exact-training (exact inference: github.com/novain-shibu/exact-llm) — 19 November 2026

## In one paragraph

Floating-point AI is not reproducible: two honest training runs that differ only in the order numbers are added become different
models (in our 3.3M-parameter test, 14% of next-token predictions differ after 6,000 steps, with validation loss within 0.3%). We built
an integer program for transformers in which every sum is exact, so training, fine-tuning, unlearning and reinforcement learning give
bit-identical weights and probabilities on every machine we tested (Intel CPUs and an NVIDIA GPU) and, by design, on any hardware that
runs the same integer program. This makes checks with zero tolerance possible: a verifier that recomputes accepts honest work with no
false alarms and rejects any change, however small.

## What we measured (each test's pass line was written before it ran; the reproduce script is public)

| Claim | Result |
|---|---|
| Honest float32 runs diverge | 0.3% / 2.4% / 10.6% / 14.1% of top-1 predictions differ at 1k / 2k / 4k / 6k steps, validation loss within 0.3% (exact pair: 0 differences and identical weights, measured at 2k steps) |
| Exact training tracks float | +0.03% (3.3M, 2k steps) to +0.64% (0.46M, 5k steps) validation loss; Lion -0.06% (0.47M, 3k steps), -0.13% (3.3M, 3k steps), +0.44% (19M, 2k steps) |
| Same bits on every device tested | numpy CPU = torch CUDA (NVIDIA T4) = torch CPU weight hashes (SGD, Lion); RL rollouts on Intel CPU and NVIDIA GPU integer-equal to the trainer |
| Real LLM | exact rank-16 fine-tune of Qwen2.5-0.5B (-6.4% val loss, +0.36% vs float), CPU = T4 bits |
| Verifiable unlearning | removing a document by replay = bit-identical to retraining without it, checked on another device; fakes rejected (toy + real Qwen adapter) |
| Training audit | 1,000 honest steps: 0 false alarms; every tampered step caught; float at zero tolerance: 100% false alarms |
| Zero-mismatch RL | 19,200 rollouts from 3 different arithmetic paths: probabilities integer-equal to the trainer's (k3 KL = 0) |
| RL swarm with cheating workers | exact verifier: 0 of 4,842 cheating rollouts accepted, 0 honest rollouts rejected; a float verifier with honestly calibrated tolerance accepted 61-65% of cheats and rejected 0.7-1.3% of honest work |
| Speed | exact training on int8 tensor cores, bit-identical to the float64 path: 5.1x (d=2048) / 6.9x (d=1024) a float32 step on a T4; with torch.compile (same bits) 2.9x (d=2048) / 4.1x (d=1024) compiled float32 |
| Your machine = ours | `python reproduce.py` (numpy only, ~3 min): exact hash ced279b58bc72962 on a Windows laptop (Intel Core Ultra, numpy 2.5.3, Python 3.12) and a Linux cloud CPU (Intel Xeon, numpy 2.1.3, Python 3.13), with default or 1 BLAS thread; float32 differs run to run, machine to machine and with the thread count |

## What it is not (limits, stated up front)

- Small scale: training from scratch up to 19M parameters; real-LLM training is a rank-16 adapter, not a full fine-tune.
- Hardware tested: Intel x86 CPUs (laptop + cloud) and NVIDIA T4. Google TPU, AMD and Apple not yet tested.
- Exact training is slower than float (2.9-4.1x compiled float32 on a T4 with int8 tensor cores + torch.compile); exact decode is
  1.04-1.26x fp16 on T4, exact prefill ~3.5x. The cost buys something float cannot give at any tolerance: zero-tolerance checks of
  someone else's work (auditing training steps, accepting RL rollouts from untrusted workers, proving a document was unlearned). In RL
  only the rollout forward pass needs to be exact on both sides, so rollouts run at exact-inference speed, not exact-training speed.
- Exact RL learns as well as float RL on its training problems (no cheater, 25,600 rollouts each: 0.36 -> 0.600 exact vs 0.587 float;
  exact rejected 0 honest rollouts, float's calibrated tolerance rejected 251, 0.98%).
- Neither exact nor float RL improved on held-out problems in our small adapter-only test (held-out p(correct) 0.29 -> 0.14 exact,
  0.29 -> 0.23 float). These two numbers are not a like-for-like comparison: that float run trained on different data, because its
  tolerance accepted cheating workers' rollouts. Float barely moved held-out accuracy at any learning rate we tried either (best
  0.31 vs 0.29 at the start), which points to the task and the small last-layer adapter rather than to integer arithmetic. A fair
  held-out comparison with no cheater has not been run yet.
- The butterfly effect itself is known (Summers & Dinneen 2021); our contribution is the exact-integer contrast.

## Related work (credit)

Bit-reproducible float training: Gensyn RepOps/Verde, OPEN-1B (Donaghy et al., arXiv 2609.17380); Srivastava, Arora & Boneh (NeurIPS 2024).
Exact integer inference: samebit, bitexact-llm, tritify (Karim, arXiv 2611.13730), Sable (Solberg, arXiv 2611.14083), CommitLLM. Same-hardware
zero-mismatch RL: vLLM / Inferact zero-mismatch-rl, vime x RL-Kernel (AMD), Zhang et al. (arXiv 2511.17826), IsoExec, Zhong et al. (arXiv 2605.14220).
Bitwise-reproducible inference across NVIDIA GPU architectures: Cooper et al. (arXiv 2609.25624). Integer-only training: Ghaffari et al. 2022, NITI.
Cross-kernel INT8 divergence: Chen (arXiv 2608.13756, arXiv 2609.00363).

## Reproduce it yourself

Run `python reproduce.py`. It trains the same tiny model twice in float32 and twice exactly; the float hashes differ, and the exact
hashes are identical and must equal ced279b58bc72962 on your machine. If you get a different exact hash, please report your CPU, OS,
Python and numpy versions.

How the exact path avoids machine-dependent maths: every operation in training is integer arithmetic (int64 adds, multiplies and
shifts; matmul products are exact integers below 2^53, so float64 BLAS returns them exactly in any summation order). The softmax uses
an integer table of exp(-i/256) x 2^24, computed once with Python's decimal module at 50 digits and rounded to the nearest integer, then
pinned by a checksum - no libm call. The data and starting weights come from the splitmix64 integer generator; each starting weight
uses a few IEEE-754 basic operations (division, multiplication, square root), which the standard requires to be correctly rounded on
every conforming machine.

## AI assistance

Built with an AI coding assistant; all results were run and checked by the author.
