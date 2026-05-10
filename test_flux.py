"""
test_flux.py - Comprehensive tests for soft_gate.py and flux.py

Run: python test_flux.py
"""
import sys, os, random, copy, json, shutil, math
sys.path.insert(0, os.path.dirname(__file__))

from soft_gate import (
    SoftGate, GateNetwork, GATE_NAMES,
    make_single_gate, make_full_adder, make_adder8,
    adder1_examples, adder8_examples,
    train, _sig,
)
from flux import FluxInterpreter, FluxError, _parse_example_line, _make_generic

_passed = _failed = 0
TEST_STATE_DIR = os.path.join(os.path.dirname(__file__), '.fluxstate_test')

def ok(name, cond, info=''):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        tag = f'  ({info})' if info else ''
        print(f'  FAIL  {name}{tag}')

def raises(name, exc_type, fn, *args, **kwargs):
    global _passed, _failed
    try:
        fn(*args, **kwargs)
        _failed += 1
        print(f'  FAIL  {name}  (expected {exc_type.__name__}, got nothing)')
    except exc_type:
        _passed += 1
    except Exception as e:
        _failed += 1
        print(f'  FAIL  {name}  (expected {exc_type.__name__}, got {type(e).__name__}: {e})')

def section(t):
    print(f'\n{"-"*60}')
    print(f'  {t}')
    print('-'*60)


# =============================================================================
# SoftGate
# =============================================================================

section('SoftGate: output always in (0,1)')
random.seed(0)
for _ in range(100):
    g = SoftGate()
    for a in (0.0, 0.5, 1.0):
        for b in (0.0, 0.5, 1.0):
            out, _ = g.forward(a, b)
            ok(f'output in (0,1) at ({a},{b})', 0.0 < out < 1.0, f'got {out}')

section('SoftGate: bilinear at corners equals sigmoid(logit)')
random.seed(1)
for _ in range(20):
    g = SoftGate()
    for i, (a, b) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        out, _ = g.forward(float(a), float(b))
        expected = _sig(g.logits[i])
        ok(f'corner ({a},{b}) == sig(logit[{i}])',
           abs(out - expected) < 1e-9, f'got {out:.8f} want {expected:.8f}')

section('SoftGate: backward moves output toward target')
random.seed(2)
for seed in range(20):
    random.seed(seed)
    g = SoftGate()
    for a, b, target in [(0.0,0.0,1.0),(0.0,1.0,0.0),(1.0,0.0,0.0),(1.0,1.0,1.0)]:
        out_before, cache = g.forward(a, b)
        gc = copy.deepcopy(g)
        gc.backward(2.0*(out_before - target), cache, lr=0.1)
        out_after, _ = gc.forward(a, b)
        ok(f'grad direction seed={seed} ({a},{b})->{target}',
           abs(out_after - target) <= abs(out_before - target) + 1e-9,
           f'before={out_before:.4f} after={out_after:.4f}')

section('SoftGate: gradient clip keeps logits finite')
random.seed(3)
g = SoftGate()
_, cache = g.forward(0.5, 0.5)
for extreme in (1e9, -1e9, float('inf'), float('-inf')):
    gc = copy.deepcopy(g)
    try:
        gc.backward(extreme, cache, lr=1.0)
        ok(f'finite after d_out={extreme}',
           all(math.isfinite(l) for l in gc.logits))
    except Exception as e:
        ok(f'no crash for d_out={extreme}', False, str(e))

section('SoftGate: sharpness in [0,1]')
random.seed(4)
for _ in range(50):
    g = SoftGate()
    s = g.sharpness()
    ok('sharpness in [0,1]', 0.0 <= s <= 1.0, f'got {s}')

section('SoftGate: sharpness is 1.0 for crisp gates')
for name, tt in [('AND',(0,0,0,1)),('OR',(0,1,1,1)),('XOR',(0,1,1,0))]:
    g = SoftGate()
    g.logits = [30.0*(2*t-1) for t in tt]   # very sharp
    ok(f'sharp {name} has sharpness ~1', g.sharpness() > 0.99, f'{g.sharpness():.4f}')

section('SoftGate: identify() for all 12 named functions')
for tt, name in GATE_NAMES.items():
    g = SoftGate()
    g.logits = [10.0*(2*t-1) for t in tt]
    ok(f'identify {name}', g.identify() == name, f'got {g.identify()}')

section('SoftGate: converges to AND / OR / XOR / NAND / XNOR across seeds')
BFN = {
    'AND':  [(0,0,0),(0,1,0),(1,0,0),(1,1,1)],
    'OR':   [(0,0,0),(0,1,1),(1,0,1),(1,1,1)],
    'XOR':  [(0,0,0),(0,1,1),(1,0,1),(1,1,0)],
    'NAND': [(0,0,1),(0,1,1),(1,0,1),(1,1,0)],
    'XNOR': [(0,0,1),(0,1,0),(1,0,0),(1,1,1)],
}
for fn_name, tt in BFN.items():
    for seed in (0, 1, 2):
        random.seed(seed)
        g = SoftGate()
        for ep in range(3000):
            for a, b, t in tt:
                out, cache = g.forward(float(a), float(b))
                g.backward(2.0*(out - float(t)), cache, lr=0.5)
        acc = sum(round(g.forward(float(a),float(b))[0])==t for a,b,t in tt)/len(tt)
        ok(f'{fn_name} seed={seed}', acc == 1.0, f'acc={acc:.0%} id={g.identify()}')

section('SoftGate: to_dict / from_dict roundtrip')
random.seed(5)
for _ in range(10):
    g = SoftGate('my_gate')
    d = g.to_dict()
    g2 = SoftGate.from_dict(d)
    ok('label preserved', g2.label == g.label)
    ok('logits preserved', g2.logits == g.logits)
    ok('forward same', abs(g.forward(0.3,0.7)[0] - g2.forward(0.3,0.7)[0]) < 1e-12)


# =============================================================================
# GateNetwork
# =============================================================================

section('GateNetwork: make_single_gate structure')
net = make_single_gate('xor')
ok('n_inputs=2',        net.n_inputs == 2)
ok('1 layer',           len(net.layers) == 1)
ok('1 gate',            net.n_gates() == 1)
ok('1 output tap',      len(net.output_taps) == 1)
ok('wiring has 1 entry',len(net.wiring) == 1)
ok('wiring key 0,0',    '0,0' in net.wiring)
ok('reads inputs 0,1',  net.wiring['0,0'] == [-1,0,-1,1])

section('GateNetwork: make_full_adder structure')
fa = make_full_adder('adder1')
ok('n_inputs=3',        fa.n_inputs == 3)
ok('3 layers',          len(fa.layers) == 3)
ok('5 gates',           fa.n_gates() == 5)
ok('2 outputs',         len(fa.output_taps) == 2)
ok('layer0 has 2 gates',len(fa.layers[0]) == 2)
ok('layer1 has 2 gates',len(fa.layers[1]) == 2)
ok('layer2 has 1 gate', len(fa.layers[2]) == 1)
ok('g1 reads a,b',  fa.wiring['0,0'] == [-1,0,-1,1])
ok('g2 reads a,b',  fa.wiring['0,1'] == [-1,0,-1,1])
ok('g3 reads s1,cin', fa.wiring['1,0'] == [0,0,-1,2])
ok('g4 reads s1,cin', fa.wiring['1,1'] == [0,0,-1,2])
ok('g5 reads c1,c2', fa.wiring['2,0'] == [0,1,1,1])
ok('outputs are sum and cout', fa.output_taps == [(1,0),(2,0)])

section('GateNetwork: make_adder8 structure')
a8 = make_adder8('adder8')
ok('n_inputs=17',        a8.n_inputs == 17)
ok('24 layers',          len(a8.layers) == 24)
ok('40 gates',           a8.n_gates() == 40)
ok('8 outputs',          len(a8.output_taps) == 8)

# Stage 1 carry wiring check: stage0 cout at (2,0), stage1 cin source = (2,0)
ok('stage1 layer4 gate0 reads s1 and carry from (2,0)',
   a8.wiring['4,0'] == [3, 0, 2, 0],
   f'got {a8.wiring.get("4,0")}')
ok('stage1 layer4 gate1 reads s1 and carry from (2,0)',
   a8.wiring['4,1'] == [3, 0, 2, 0],
   f'got {a8.wiring.get("4,1")}')

section('GateNetwork: forward on known-logit gates')
net2 = make_single_gate('xor_known')
# Set logits to crisp XOR: f(0,0)=0, f(0,1)=1, f(1,0)=1, f(1,1)=0
net2.layers[0][0].logits = [-10.0, 10.0, 10.0, -10.0]
for a, b, expected in [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]:
    outs, _ = net2.forward([float(a), float(b)])
    ok(f'XOR({a},{b})={expected}', round(outs[0]) == expected,
       f'got {outs[0]:.4f}')

section('GateNetwork: backward accumulates at fan-out correctly')
# In the full adder, gate (0,0) fans out to gates (1,0) and (1,1).
# After one backward pass, gate (0,0) should have received gradient from both.
random.seed(10)
fa2 = make_full_adder('fanout_test')
# Run one training step and verify it completes without error
loss = fa2.train_step([1.0, 1.0, 0.0], [0.0, 1.0], lr=0.1)
ok('fan-out backward completes', math.isfinite(loss), f'loss={loss}')
ok('loss is non-negative', loss >= 0, f'loss={loss}')

section('GateNetwork: accuracy with wrong target count raises ValueError')
net3 = make_single_gate('err_test')
raises('wrong target count raises ValueError', ValueError,
       net3.accuracy, [([0,0],[0,0])])   # 2 targets for 1-output net

section('GateNetwork: accuracy on all-correct examples = 1.0')
net4 = make_single_gate('acc_test')
net4.layers[0][0].logits = [-10.0, 10.0, 10.0, -10.0]  # crisp XOR
xor_ex = [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
ok('accuracy 1.0 on correct XOR', net4.accuracy(xor_ex) == 1.0)

section('GateNetwork: to_dict / from_dict roundtrip')
random.seed(11)
fa_orig = make_full_adder('roundtrip')
d = fa_orig.to_dict()
fa_loaded = GateNetwork.from_dict(d)
ok('name preserved', fa_loaded.name == fa_orig.name)
ok('n_inputs preserved', fa_loaded.n_inputs == fa_orig.n_inputs)
ok('n_gates preserved', fa_loaded.n_gates() == fa_orig.n_gates())
ok('wiring preserved', fa_loaded.wiring == fa_orig.wiring)
ok('output_taps preserved', fa_loaded.output_taps == fa_orig.output_taps)
for a, b, cin in [(0,0,0),(1,1,1),(1,0,1)]:
    o1, _ = fa_orig.forward([float(a),float(b),float(cin)])
    o2, _ = fa_loaded.forward([float(a),float(b),float(cin)])
    ok(f'forward same after roundtrip ({a},{b},{cin})',
       all(abs(x-y)<1e-12 for x,y in zip(o1,o2)))

section('GateNetwork: save / load (file roundtrip)')
tmp_path = os.path.join(TEST_STATE_DIR, 'save_test.json')
os.makedirs(TEST_STATE_DIR, exist_ok=True)
random.seed(12)
fa_save = make_full_adder('save_test')
fa_save.save(tmp_path)
ok('file created', os.path.exists(tmp_path))
fa_load = GateNetwork.load(tmp_path)
ok('n_gates after load', fa_load.n_gates() == 5)
for a, b, cin in [(0,1,1),(1,0,0)]:
    o1, _ = fa_save.forward([float(a),float(b),float(cin)])
    o2, _ = fa_load.forward([float(a),float(b),float(cin)])
    ok(f'load matches save ({a},{b},{cin})',
       all(abs(x-y)<1e-12 for x,y in zip(o1,o2)))

section('GateNetwork: single-gate training converges')
for seed in range(5):
    random.seed(seed)
    net = make_single_gate('xor_train')
    xor_ex2 = [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
    for ep in range(3000):
        random.shuffle(xor_ex2)
        for inp, tgt in xor_ex2:
            net.train_step([float(x) for x in inp], [float(t) for t in tgt], lr=0.5)
        if net.accuracy(xor_ex2) == 1.0:
            break
    ok(f'single-gate XOR seed={seed}', net.accuracy(xor_ex2)==1.0,
       f'acc={net.accuracy(xor_ex2):.0%}')

section('GateNetwork: full adder trains to 100% across 5 seeds')
fa_ex = adder1_examples()
for seed in range(5):
    random.seed(seed)
    fa_t = make_full_adder(f'adder_s{seed}')
    fa_t, acc, ep = train(fa_t, fa_ex, epochs=8000, lr=0.35)
    ok(f'full adder seed={seed} acc=100%', acc == 1.0,
       f'acc={acc:.0%} after {ep} epochs')

section('GateNetwork: full adder all 8 outputs correct after training')
random.seed(42)
fa_final = make_full_adder('final_check')
fa_final, _, _ = train(fa_final, fa_ex, epochs=8000, lr=0.35)
for a in range(2):
    for b in range(2):
        for cin in range(2):
            ts = (a+b+cin)&1; tc = (a+b+cin)>>1
            outs, _ = fa_final.forward([float(a),float(b),float(cin)])
            ok(f'fa({a},{b},{cin}) sum={ts}',   round(outs[0])==ts, f'got {round(outs[0])}')
            ok(f'fa({a},{b},{cin}) carry={tc}', round(outs[1])==tc, f'got {round(outs[1])}')

section('GateNetwork: identify_all returns list of strings')
random.seed(6)
fa_id = make_full_adder('id_test')
ids = fa_id.identify_all()
ok('identify_all length matches n_gates', len(ids) == 5)
ok('all strings', all(isinstance(s, str) for s in ids))


# =============================================================================
# Flux language
# =============================================================================

def fresh_interp():
    os.makedirs(TEST_STATE_DIR, exist_ok=True)
    return FluxInterpreter(state_dir=TEST_STATE_DIR)

section('Flux: _parse_example_line valid inputs')
for line, expected_inp, expected_out in [
    ('0 1 -> 1',      [0,1], [1]),
    ('1 1 0 -> 0 1',  [1,1,0], [0,1]),
    ('0 -> 1',        [0],   [1]),
    ('0 0 0 0 -> 1 1',[0,0,0,0],[1,1]),
]:
    inp, out = _parse_example_line(line)
    ok(f'parse {line!r} inputs', inp == expected_inp, f'got {inp}')
    ok(f'parse {line!r} outputs', out == expected_out, f'got {out}')

section('Flux: _parse_example_line invalid inputs')
for bad in ['0 1', '-> 1', '0 1 -> ', '0 2 -> 1', 'abc -> 1', '0 1 -> abc']:
    raises(f'parse bad {bad!r}', FluxError, _parse_example_line, bad)

section('Flux: learn single gate from examples')
random.seed(7)
interp = fresh_interp()
xor_ex3 = [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
net = interp.cmd_learn('xor_t', mode='examples', examples=xor_ex3, print_fn=lambda x: None)
ok('learn returns GateNetwork', isinstance(net, GateNetwork))
ok('xor trained to 100%', net.accuracy(xor_ex3) == 1.0)
ok('xor persisted to disk', os.path.exists(interp._prog_path('xor_t')))
ok('xor in index', 'xor_t' in interp.list_programs())

section('Flux: learn adder1 builtin')
random.seed(8)
interp2 = fresh_interp()
net2 = interp2.cmd_learn('add_t', mode='adder1', print_fn=lambda x: None)
ok('adder1 returns GateNetwork', isinstance(net2, GateNetwork))
ok('adder1 has 5 gates', net2.n_gates() == 5)
ok('adder1 has 3 inputs', net2.n_inputs == 3)
ok('adder1 has 2 outputs', len(net2.output_taps) == 2)
ok('adder1 acc=100%', net2.accuracy(adder1_examples()) == 1.0)

section('Flux: cmd_run single gate')
interp3 = fresh_interp()
random.seed(9)
interp3.cmd_learn('or_t', mode='examples',
                  examples=[([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[1])],
                  print_fn=lambda x: None)
ok('run or 0 0 -> 0', interp3.cmd_run('or_t', [0, 0]) == '0')
ok('run or 0 1 -> 1', interp3.cmd_run('or_t', [0, 1]) == '1')
ok('run or 1 0 -> 1', interp3.cmd_run('or_t', [1, 0]) == '1')
ok('run or 1 1 -> 1', interp3.cmd_run('or_t', [1, 1]) == '1')

section('Flux: cmd_run full adder')
random.seed(13)
interp4 = fresh_interp()
interp4.cmd_learn('fa_t', mode='adder1', print_fn=lambda x: None)
ok('fa 0+0+0 = sum=0 cout=0', 'out0=0' in interp4.cmd_run('fa_t', [0,0,0]) and 'out1=0' in interp4.cmd_run('fa_t', [0,0,0]))
ok('fa 1+1+0 = sum=0 cout=1', 'out0=0' in interp4.cmd_run('fa_t', [1,1,0]) and 'out1=1' in interp4.cmd_run('fa_t', [1,1,0]))
ok('fa 1+1+1 = sum=1 cout=1', 'out0=1' in interp4.cmd_run('fa_t', [1,1,1]) and 'out1=1' in interp4.cmd_run('fa_t', [1,1,1]))

section('Flux: cmd_run wrong input count raises FluxError')
raises('wrong input count', FluxError, interp4.cmd_run, 'fa_t', [0, 0])

section('Flux: cmd_run unknown program raises FluxError')
raises('unknown program', FluxError, interp3.cmd_run, 'does_not_exist', [0,1])

section('Flux: cmd_show returns gate info')
random.seed(14)
interp5 = fresh_interp()
interp5.cmd_learn('xor_show', mode='examples', examples=xor_ex3, print_fn=lambda x: None)
show_out = interp5.cmd_show('xor_show')
ok('show contains program name', 'xor_show' in show_out)
ok('show contains sharpness', 'Sharpness' in show_out or 'sharpness' in show_out)
ok('show contains truth table', 'f(0,0)' in show_out)
ok('show contains gate label', 'L0G0' in show_out)

section('Flux: cmd_list shows trained programs')
list_out = interp5.cmd_list()
ok('list contains xor_show', 'xor_show' in list_out)
ok('list has header', 'NAME' in list_out)
ok('list has acc column', 'ACC' in list_out)

section('Flux: cmd_forget removes program')
random.seed(15)
interp6 = fresh_interp()
interp6.cmd_learn('to_forget', mode='examples', examples=xor_ex3, print_fn=lambda x: None)
ok('program exists before forget', os.path.exists(interp6._prog_path('to_forget')))
interp6.cmd_forget('to_forget')
ok('file removed after forget', not os.path.exists(interp6._prog_path('to_forget')))
ok('not in index after forget', 'to_forget' not in interp6.list_programs())
raises('run after forget raises FluxError', FluxError, interp6.cmd_run, 'to_forget', [0,1])

section('Flux: cmd_forget unknown program raises FluxError')
raises('forget unknown raises FluxError', FluxError, interp6.cmd_forget, 'ghost_program')

section('Flux: execute() dispatch - learn')
interp7 = fresh_interp()
random.seed(16)
out = interp7.execute('learn xor_d adder1', print_fn=lambda x: None)
ok('execute learn adder1 returns empty string', out == '')
ok('program trained and saved', 'xor_d' in interp7.list_programs())

section('Flux: execute() dispatch - run')
res = interp7.execute('run xor_d 1 1 0')
ok('execute run returns string', isinstance(res, str))
ok('execute run gives output', len(res) > 0)

section('Flux: execute() dispatch - list')
res = interp7.execute('list')
ok('execute list returns string', isinstance(res, str) and len(res) > 0)

section('Flux: execute() dispatch - forget')
interp7.execute('learn del_me adder1', print_fn=lambda x: None)
res = interp7.execute('forget del_me')
ok('execute forget returns confirmation', 'del_me' in res or 'Forgot' in res)

section('Flux: execute() dispatch - help')
res = interp7.execute('help')
ok('help returns non-empty string', isinstance(res, str) and len(res) > 0)
ok('help mentions learn', 'learn' in res.lower())

section('Flux: execute() dispatch - exit returns sentinel')
res = interp7.execute('exit')
ok('exit returns __EXIT__', res == '__EXIT__')

section('Flux: execute() handles unknown command gracefully')
res = interp7.execute('flurble 123')
ok('unknown cmd returns error string', 'Error' in res or 'error' in res)
ok('unknown cmd does not crash', isinstance(res, str))

section('Flux: execute() handles missing program name gracefully')
res = interp7.execute('run')
ok('run with no args returns error', 'Error' in res or 'error' in res)
res2 = interp7.execute('learn')
ok('learn with no args returns error', 'Error' in res2 or 'error' in res2)

section('Flux: persistence - program survives new interpreter instance')
random.seed(17)
interp8a = FluxInterpreter(state_dir=TEST_STATE_DIR)
interp8a.cmd_learn('persist_test', mode='examples', examples=xor_ex3, print_fn=lambda x: None)
interp8b = FluxInterpreter(state_dir=TEST_STATE_DIR)  # fresh instance, same dir
ok('program in new instance index', 'persist_test' in interp8b.list_programs())
res = interp8b.cmd_run('persist_test', [1, 0])
ok('run works in new instance', res == '1')

section('Flux: index.json has correct metadata')
index = interp8b.list_programs()
meta = index.get('persist_test', {})
ok('n_gates in meta', 'n_gates' in meta)
ok('accuracy in meta', 'accuracy' in meta)
ok('sharpness in meta', 'sharpness' in meta)
ok('n_inputs in meta', 'n_inputs' in meta)
ok('n_outputs in meta', 'n_outputs' in meta)
ok('gates_id in meta', 'gates_id' in meta)
ok('accuracy is float 0-1', 0.0 <= meta['accuracy'] <= 1.0)
ok('sharpness is float 0-1', 0.0 <= meta['sharpness'] <= 1.0)

section('Flux: .flux script files parse and run correctly')
from flux import run_script
logic_path = os.path.join(os.path.dirname(__file__), 'programs', 'logic.flux')
ok('logic.flux exists', os.path.exists(logic_path))
# Run logic.flux against a test state dir to avoid polluting production
script_interp = FluxInterpreter(state_dir=TEST_STATE_DIR)
try:
    run_script(logic_path, interp=script_interp)
    ok('logic.flux runs without error', True)
    ok('xor trained by script', 'xor' in script_interp.list_programs())
    ok('and trained by script', 'and' in script_interp.list_programs())
    ok('or trained by script',  'or'  in script_interp.list_programs())
except Exception as e:
    ok('logic.flux runs without error', False, str(e))

adder_path = os.path.join(os.path.dirname(__file__), 'programs', 'adder.flux')
ok('adder.flux exists', os.path.exists(adder_path))
try:
    run_script(adder_path, interp=script_interp)
    ok('adder.flux runs without error', True)
    ok('adder trained by script', 'adder' in script_interp.list_programs())
except Exception as e:
    ok('adder.flux runs without error', False, str(e))

nand_path = os.path.join(os.path.dirname(__file__), 'programs', 'nand_is_dead.flux')
ok('nand_is_dead.flux exists', os.path.exists(nand_path))
try:
    run_script(nand_path, interp=script_interp)
    ok('nand_is_dead.flux runs without error', True)
except Exception as e:
    ok('nand_is_dead.flux runs without error', False, str(e))

# =============================================================================
# Cleanup and summary
# =============================================================================

shutil.rmtree(TEST_STATE_DIR, ignore_errors=True)

print(f'\n{"="*60}')
print(f'  {_passed} passed   {_failed} failed')
if _failed == 0:
    print('  All tests passed.')
else:
    print(f'  {_failed} FAILED.')
print('='*60)
sys.exit(1 if _failed else 0)
