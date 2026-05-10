"""
test_diff_cpu.py - Thorough test suite for diff_cpu.py

Run: python test_diff_cpu.py
"""
import sys, random, copy, math
sys.path.insert(0, '.')

from diff_cpu import (
    soft_not, soft_and, soft_or, soft_xor, soft_nand,
    _soft_ha, _soft_fa, soft_add8,
    Gate, LearnableFullAdder, _sig,
)
from alu import add8

_passed = _failed = 0

def ok(name, cond, info=''):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        tag = f'  ({info})' if info else ''
        print(f'  FAIL  {name}{tag}')

def section(t):
    print(f'\n{"-"*58}')
    print(f'  {t}')
    print('-'*58)

# ── Soft gate truth tables on {0,1} ──────────────────────────────
section('Soft gates: exact on all {0,1} corners')

for a in (0, 1):
    ok(f'soft_not({a})',  soft_not(float(a))  == float(1 - a))

for a in (0, 1):
    for b in (0, 1):
        exp_and  = float(a & b)
        exp_or   = float(a | b)
        exp_xor  = float(a ^ b)
        exp_nand = float(0 if (a and b) else 1)
        ok(f'soft_and({a},{b})',  soft_and(float(a),float(b))  == exp_and)
        ok(f'soft_or({a},{b})',   soft_or(float(a),float(b))   == exp_or)
        ok(f'soft_xor({a},{b})',  soft_xor(float(a),float(b))  == exp_xor)
        ok(f'soft_nand({a},{b})', soft_nand(float(a),float(b)) == exp_nand)

# ── Soft gates: values in [0,1] for soft inputs ──────────────────
section('Soft gates: output stays in [0,1] for soft inputs')

for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    ok(f'soft_not({a}) in [0,1]',    0.0 <= soft_not(a) <= 1.0)
    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        ok(f'soft_and({a},{b}) in [0,1]',  0.0 <= soft_and(a,b)  <= 1.0)
        ok(f'soft_or({a},{b}) in [0,1]',   0.0 <= soft_or(a,b)   <= 1.0)
        ok(f'soft_xor({a},{b}) in [0,1]',  0.0 <= soft_xor(a,b)  <= 1.0)
        ok(f'soft_nand({a},{b}) in [0,1]', 0.0 <= soft_nand(a,b) <= 1.0)

# ── Half adder truth table ────────────────────────────────────────
section('Soft half adder: all {0,1} corners')

for a in (0, 1):
    for b in (0, 1):
        s, c = _soft_ha(float(a), float(b))
        ok(f'ha({a},{b}) sum={a^b}',   round(s) == (a ^ b), f'got {s:.4f}')
        ok(f'ha({a},{b}) carry={a&b}', round(c) == (a & b), f'got {c:.4f}')

# ── Full adder truth table ────────────────────────────────────────
section('Soft full adder: all {0,1} corners')

for a in (0, 1):
    for b in (0, 1):
        for cin in (0, 1):
            s2, co = _soft_fa(float(a), float(b), float(cin))
            tot = a + b + cin
            ok(f'fa({a},{b},{cin}) sum={tot&1}',   round(s2) == (tot&1),  f'got {s2:.4f}')
            ok(f'fa({a},{b},{cin}) carry={tot>>1}', round(co) == (tot>>1), f'got {co:.4f}')

# ── soft_add8 boundary cases ──────────────────────────────────────
section('soft_add8: boundary cases vs alu.add8')

boundaries = [
    (0,   0),   # zero + zero
    (255, 0),   # max + zero
    (0,   255), # zero + max
    (255, 255), # overflow wrap
    (127, 1),   # sign boundary
    (128, 128), # double sign boundary
    (1,   1),   # small
    (100, 156), # sum = 256, wraps to 0
    (200, 56),  # sum in range
    (15,  240), # sum = 255, no wrap
]
for a, b in boundaries:
    sv, sc = soft_add8(a, b)
    hv, hc = add8(a, b)
    ok(f'soft_add8({a},{b}) val',   sv == hv, f'got {sv} want {hv}')
    ok(f'soft_add8({a},{b}) carry', sc == hc, f'got {sc} want {hc}')

# ── soft_add8 random 1000 pairs ───────────────────────────────────
section('soft_add8 vs alu.add8: 1000 random pairs')

random.seed(0)
for _ in range(1000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    sv, sc = soft_add8(a, b)
    hv, hc = add8(a, b)
    ok(f'soft_add8({a},{b})', sv == hv and sc == hc,
       f'soft=({sv},{sc}) hard=({hv},{hc})')

# ── Gate forward: output strictly in (0,1) ───────────────────────
section('Gate.fwd: output always strictly in (0,1)')

random.seed(1)
for _ in range(50):
    g = Gate('range_test')
    for a in (0.0, 0.5, 1.0):
        for b in (0.0, 0.5, 1.0):
            out, _ = g.fwd(a, b)
            ok(f'gate output in (0,1) at ({a},{b})',
               0.0 < out < 1.0, f'got {out}')

# ── Gate: forward is bilinear interpolation ───────────────────────
section('Gate.fwd: bilinear interpolation property')

random.seed(2)
g = Gate('bilinear')
# At exactly (0,0), (0,1), (1,0), (1,1) the output must equal sigmoid(logit)
for i, (a, b) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
    out, _ = g.fwd(float(a), float(b))
    expected = _sig(g.logits[i])
    ok(f'Gate bilinear at ({a},{b}) == sig(logit[{i}])',
       abs(out - expected) < 1e-9, f'got {out:.6f} want {expected:.6f}')

# ── Gate: gradient moves output toward target ─────────────────────
section('Gate.bwd: output moves toward target after one step')

for seed in range(15):
    random.seed(seed)
    g = Gate('gdir')
    for a_val, b_val, target in [(0.0,0.0,1.0),(0.0,1.0,0.0),(1.0,0.0,0.0),(1.0,1.0,1.0)]:
        out_before, cache = g.fwd(a_val, b_val)
        g_copy = copy.deepcopy(g)
        g_copy.bwd(2.0 * (out_before - target), cache, lr=0.1)
        out_after, _ = g_copy.fwd(a_val, b_val)
        ok(f'gradient toward target seed={seed} ({a_val},{b_val})->{target}',
           abs(out_after - target) <= abs(out_before - target) + 1e-9,
           f'before={out_before:.4f} after={out_after:.4f}')

# ── Gate: gradient clip prevents explosion ────────────────────────
section('Gate.bwd: extreme d_out is clipped; logits stay finite')

random.seed(3)
g_clip = Gate('clip_test')
orig_logits = list(g_clip.logits)
_, cache = g_clip.fwd(0.5, 0.5)
for extreme in (1e6, -1e6, 1e12, float('inf')):
    g_test = copy.deepcopy(g_clip)
    try:
        g_test.bwd(extreme, cache, lr=1.0)
        all_finite = all(math.isfinite(l) for l in g_test.logits)
        ok(f'logits finite after d_out={extreme}', all_finite,
           str(g_test.logits))
    except Exception as e:
        ok(f'no exception for d_out={extreme}', False, str(e))

# ── Gate converges to each common boolean function ────────────────
section('Gate: converges to AND / OR / XOR / NAND / NOR / XNOR')

BOOL_FNS = {
    'AND':  [(0,0,0),(0,1,0),(1,0,0),(1,1,1)],
    'OR':   [(0,0,0),(0,1,1),(1,0,1),(1,1,1)],
    'XOR':  [(0,0,0),(0,1,1),(1,0,1),(1,1,0)],
    'NAND': [(0,0,1),(0,1,1),(1,0,1),(1,1,0)],
    'NOR':  [(0,0,1),(0,1,0),(1,0,0),(1,1,0)],
    'XNOR': [(0,0,1),(0,1,0),(1,0,0),(1,1,1)],
}
for fn_name, tt in BOOL_FNS.items():
    for seed in (0, 1, 2):
        random.seed(seed)
        g = Gate(fn_name)
        for ep in range(3000):
            random.shuffle(tt)
            for a, b, t in tt:
                out, cache = g.fwd(float(a), float(b))
                g.bwd(2.0 * (out - float(t)), cache, lr=0.5)
        acc = sum(round(g.fwd(float(a),float(b))[0]) == t for a,b,t in tt) / len(tt)
        ok(f'Gate learns {fn_name} (seed={seed})', acc == 1.0,
           f'acc={acc:.0%}  identified={g.identify()}')

# ── LearnableFullAdder: convergence across 5 seeds ───────────────
section('LearnableFullAdder: converges to 100% across 5 seeds')

examples_fa = [
    (a, b, cin, (a+b+cin)&1, (a+b+cin)>>1)
    for a in range(2) for b in range(2) for cin in range(2)
]
for seed in range(5):
    random.seed(seed)
    fa = LearnableFullAdder()
    converged = False
    for ep in range(8000):
        random.shuffle(examples_fa)
        for a, b, cin, ts, tc in examples_fa:
            fa.step(float(a), float(b), float(cin), float(ts), float(tc), lr=0.4)
        if fa.accuracy() == 1.0:
            converged = True
            break
    ok(f'LearnableFullAdder seed={seed} converges',
       converged, f'acc={fa.accuracy():.0%} after {ep+1} epochs')

# ── LearnableFullAdder: all 8 outputs correct after training ──────
section('LearnableFullAdder: all 8 (a,b,cin) -> (sum,carry) correct')

random.seed(42)
fa2 = LearnableFullAdder()
for ep in range(6000):
    random.shuffle(examples_fa)
    for a, b, cin, ts, tc in examples_fa:
        fa2.step(float(a), float(b), float(cin), float(ts), float(tc), lr=0.4)

for a in range(2):
    for b in range(2):
        for cin in range(2):
            ts = (a+b+cin) & 1
            tc = (a+b+cin) >> 1
            s2, co, _ = fa2.fwd(float(a), float(b), float(cin))
            ok(f'fa2({a},{b},{cin}) sum',   round(s2) == ts, f'got {round(s2)} want {ts}')
            ok(f'fa2({a},{b},{cin}) carry', round(co) == tc, f'got {round(co)} want {tc}')

# ── LearnableFullAdder: gate functions are functionally valid ─────
section('LearnableFullAdder: gate functions implement valid adder circuit')

# After training, gates g1/g3 must implement XOR or XNOR (which also works),
# g2/g4 must implement AND or an equivalent, g5 must be OR or equivalent.
# We verify by exhaustive truth table check rather than name matching.

def gate_truth_table(g):
    return [(round(g.fwd(float(a), float(b))[0])) for a in range(2) for b in range(2)]

s1_is_xor_or_xnor = gate_truth_table(fa2.g1) in [[0,1,1,0],[1,0,0,1]]
ok('g1 implements XOR or XNOR', s1_is_xor_or_xnor,
   f'got {gate_truth_table(fa2.g1)} ({fa2.g1.identify()})')

s2_is_xor_or_xnor = gate_truth_table(fa2.g3) in [[0,1,1,0],[1,0,0,1]]
ok('g3 implements XOR or XNOR', s2_is_xor_or_xnor,
   f'got {gate_truth_table(fa2.g3)} ({fa2.g3.identify()})')

# ── Summary ───────────────────────────────────────────────────────
print(f'\n{"="*58}')
print(f'  {_passed} passed   {_failed} failed')
if _failed == 0:
    print('  All tests passed.')
else:
    print(f'  {_failed} FAILED.')
print('='*58)
sys.exit(1 if _failed else 0)
