"""
soft_gate.py -- The new primitive: one gate type that learns any boolean function.

Instead of a zoo of fixed gates (NAND, AND, OR, XOR ...), FluxOS has one
unit: the SoftGate. It starts random and converges to whatever function
gradient descent finds useful. NAND, XOR, OR -- all emerge from training.

GateNetwork chains SoftGates into trainable programs. Topology is explicit
(you specify which gate feeds which), so the structure is fixed but the
logic inside every gate is learned. Programs are saved as JSON; loading a
program restores the exact trained state.

No native math is used for computation -- bilinear interpolation over the
truth table uses only + - * operations (polynomial arithmetic, same
philosophy as soft_nand = 1 - a*b in diff_cpu.py). math.exp is used only
for the sigmoid needed by the learning machinery.
"""

import math
import json
import os
import random
from typing import Optional

# ── Sigmoid (only transcendental; needed for learning, not computation) ───────

def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

# ── Gate identity table ───────────────────────────────────────────────────────
# Key: (f(0,0), f(0,1), f(1,0), f(1,1)) where first arg = a, second = b.

GATE_NAMES: dict[tuple, str] = {
    (0, 0, 0, 0): 'FALSE',
    (1, 1, 1, 1): 'TRUE',
    (1, 1, 1, 0): 'NAND',
    (0, 0, 0, 1): 'AND',
    (0, 1, 1, 0): 'XOR',
    (1, 0, 0, 1): 'XNOR',
    (0, 1, 1, 1): 'OR',
    (1, 0, 0, 0): 'NOR',
    (1, 1, 0, 0): 'NOT_A',
    (0, 0, 1, 1): 'A',
    (1, 0, 1, 0): 'NOT_B',
    (0, 1, 0, 1): 'B',
    (0, 0, 1, 0): 'A_NOT_B',
    (1, 1, 0, 1): 'A_OR_NB',
    (0, 1, 0, 0): 'B_NOT_A',
    (1, 0, 1, 1): 'B_OR_NA',
}

# ── SoftGate ──────────────────────────────────────────────────────────────────

class SoftGate:
    """
    A single learnable gate: 4 logits -> any 2-input boolean function.

    Forward pass: bilinear interpolation over sigmoid-activated truth table.
      w = [(1-a)(1-b), (1-a)b, a(1-b), ab]   (barycentric coords)
      out = sum(w[i] * sigmoid(logits[i]))

    Backward pass: analytic chain-rule gradients (d_out -> d_a, d_b),
    logits updated in-place by SGD.

    At inference time, round output to get hard 0/1. At training time,
    keep soft for gradient flow.
    """

    def __init__(self, label: str = 'gate', logits: Optional[list] = None):
        self.label = label
        self.logits: list[float] = (
            list(logits) if logits is not None
            else [random.gauss(0, 2.0) for _ in range(4)]
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, a: float, b: float) -> tuple[float, tuple]:
        """Returns (output, cache). cache is passed unchanged to backward()."""
        t = [_sig(l) for l in self.logits]
        w = [(1-a)*(1-b), (1-a)*b, a*(1-b), a*b]
        out = sum(w[i] * t[i] for i in range(4))
        return out, (a, b, t, w)

    # ── Backward ──────────────────────────────────────────────────────────────

    def backward(self, d_out: float, cache: tuple, lr: float) -> tuple[float, float]:
        """
        Update logits via SGD. Returns (d_a, d_b) for upstream gates.
        d_out is clipped to [-5, 5] to prevent gradient explosion.
        """
        a, b, t, w = cache
        d_out = max(-5.0, min(5.0, d_out))
        for i in range(4):
            self.logits[i] -= lr * d_out * w[i] * t[i] * (1.0 - t[i])
        dw_da = [-(1-b), -b,    (1-b), b  ]
        dw_db = [-(1-a),  (1-a), -a,   a  ]
        d_a = d_out * sum(dw_da[i] * t[i] for i in range(4))
        d_b = d_out * sum(dw_db[i] * t[i] for i in range(4))
        return d_a, d_b

    # ── Introspection ─────────────────────────────────────────────────────────

    def truth_table(self) -> list[float]:
        """Current truth table values (soft, in (0,1))."""
        return [_sig(l) for l in self.logits]

    def identify(self) -> str:
        """Name of the nearest canonical boolean function."""
        tt = tuple(round(_sig(l)) for l in self.logits)
        return GATE_NAMES.get(tt, str(list(tt)))

    def sharpness(self) -> float:
        """
        0.0 = gate undecided (all outputs ~0.5).
        1.0 = gate fully committed (all outputs near 0 or 1).
        """
        return sum(abs(_sig(l) - 0.5) * 2.0 for l in self.logits) / 4.0

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {'label': self.label, 'logits': self.logits}

    @classmethod
    def from_dict(cls, d: dict) -> 'SoftGate':
        return cls(label=d['label'], logits=d['logits'])


# ── GateNetwork ───────────────────────────────────────────────────────────────

class GateNetwork:
    """
    A directed acyclic network of SoftGates.

    Wiring: each gate (layer_i, gate_j) reads from two sources.
    Sources are specified as (src_layer, src_idx) pairs where
    src_layer == -1 means a primary input (src_idx is the input index).

    Wiring dict: str key "li,gi" -> [src_la, src_ga, src_lb, src_gb]
    (JSON requires string keys, so tuples are encoded as "li,gi".)

    Output taps: list of (layer_i, gate_j) pairs -- which gate outputs
    are the network outputs, in order.

    Forward: run all layers in order, collecting (output, cache) per gate.
    Backward: reverse layers, accumulate gradients at fan-out points.
    """

    def __init__(
        self,
        name: str,
        n_inputs: int,
        layers: list[list[SoftGate]],
        wiring: dict,
        output_taps: list[tuple],
    ):
        self.name = name
        self.n_inputs = n_inputs
        self.layers = layers
        self.wiring = wiring           # {str "li,gi": [la,ga,lb,gb]}
        self.output_taps = output_taps # [(li,gi), ...]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, inputs: list[float]) -> tuple[list[float], dict]:
        """
        Run the network. Returns (outputs, activation_cache).
        activation_cache: {(li,gi): (output_val, gate_cache)}
        """
        cache: dict[tuple, tuple] = {}
        for li, layer in enumerate(self.layers):
            for gi, gate in enumerate(layer):
                src_la, src_ga, src_lb, src_gb = self.wiring[f'{li},{gi}']
                a = inputs[src_ga] if src_la == -1 else cache[(src_la, src_ga)][0]
                b = inputs[src_gb] if src_lb == -1 else cache[(src_lb, src_gb)][0]
                out, gc = gate.forward(a, b)
                cache[(li, gi)] = (out, gc)
        outputs = [cache[tuple(tap)][0] for tap in self.output_taps]
        return outputs, cache

    # ── Backward ──────────────────────────────────────────────────────────────

    def backward(self, d_outputs: list[float], act_cache: dict, lr: float) -> None:
        """
        Backprop through the DAG. Accumulates gradients at fan-out.
        """
        grad: dict[tuple, float] = {}

        # Seed gradients at output taps
        for i, tap in enumerate(self.output_taps):
            key = tuple(tap)
            grad[key] = grad.get(key, 0.0) + d_outputs[i]

        # Reverse topological order: layers high to low, gates any order
        for li in range(len(self.layers) - 1, -1, -1):
            for gi in range(len(self.layers[li])):
                key = (li, gi)
                if key not in grad:
                    continue
                gate = self.layers[li][gi]
                _, gc = act_cache[key]
                d_a, d_b = gate.backward(grad[key], gc, lr)

                src_la, src_ga, src_lb, src_gb = self.wiring[f'{li},{gi}']
                if src_la != -1:  # upstream gate (not primary input)
                    up_key = (src_la, src_ga)
                    grad[up_key] = grad.get(up_key, 0.0) + d_a
                if src_lb != -1:
                    up_key = (src_lb, src_gb)
                    grad[up_key] = grad.get(up_key, 0.0) + d_b

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(self, inputs: list[float], targets: list[float], lr: float) -> float:
        """One forward+backward+update. Returns MSE loss."""
        outputs, act_cache = self.forward(inputs)
        loss = sum((o - t) ** 2 for o, t in zip(outputs, targets))
        d_outputs = [2.0 * (o - t) for o, t in zip(outputs, targets)]
        self.backward(d_outputs, act_cache, lr)
        return loss

    def accuracy(self, examples: list) -> float:
        """Fraction of examples where all output bits round correctly."""
        if not examples:
            return 0.0
        ok = 0
        for inputs, targets in examples:
            outs, _ = self.forward([float(x) for x in inputs])
            if all(round(o) == int(t) for o, t in zip(outs, targets)):
                ok += 1
        return ok / len(examples)

    # ── Introspection ─────────────────────────────────────────────────────────

    def sharpness(self) -> float:
        all_gates = [g for layer in self.layers for g in layer]
        if not all_gates:
            return 0.0
        return sum(g.sharpness() for g in all_gates) / len(all_gates)

    def identify_all(self) -> list[str]:
        return [g.identify() for layer in self.layers for g in layer]

    def n_gates(self) -> int:
        return sum(len(layer) for layer in self.layers)

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'name':        self.name,
            'n_inputs':    self.n_inputs,
            'layers':      [[g.to_dict() for g in layer] for layer in self.layers],
            'wiring':      self.wiring,
            'output_taps': self.output_taps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GateNetwork':
        layers = [[SoftGate.from_dict(gd) for gd in layer] for layer in d['layers']]
        return cls(
            name=d['name'],
            n_inputs=d['n_inputs'],
            layers=layers,
            wiring=d['wiring'],
            output_taps=[tuple(t) for t in d['output_taps']],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'GateNetwork':
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ── Factory: single gate (2 inputs, 1 output) ─────────────────────────────────

def make_single_gate(name: str = 'gate') -> GateNetwork:
    """One SoftGate learning any 2-input boolean function."""
    return GateNetwork(
        name=name,
        n_inputs=2,
        layers=[[SoftGate(f'{name}(a,b)->out')]],
        wiring={'0,0': [-1, 0, -1, 1]},
        output_taps=[(0, 0)],
    )

# ── Factory: 1-bit full adder (3 inputs, 2 outputs, 5 gates) ─────────────────

def make_full_adder(name: str = 'adder1') -> GateNetwork:
    """
    5-gate full adder. Same topology as LearnableFullAdder in diff_cpu.py.

    Layer 0: g1(a,b)->s1  g2(a,b)->c1
    Layer 1: g3(s1,cin)->sum  g4(s1,cin)->c2
    Layer 2: g5(c1,c2)->cout
    Outputs: [sum, cout]
    """
    return GateNetwork(
        name=name,
        n_inputs=3,
        layers=[
            [SoftGate(f'{name}.g1(a,b)->s1'),  SoftGate(f'{name}.g2(a,b)->c1')],
            [SoftGate(f'{name}.g3(s1,cin)->sum'), SoftGate(f'{name}.g4(s1,cin)->c2')],
            [SoftGate(f'{name}.g5(c1,c2)->cout')],
        ],
        wiring={
            '0,0': [-1, 0, -1, 1],   # g1: input_a, input_b
            '0,1': [-1, 0, -1, 1],   # g2: input_a, input_b
            '1,0': [0,  0, -1, 2],   # g3: s1 (layer0,gate0), cin (input_2)
            '1,1': [0,  0, -1, 2],   # g4: s1 (layer0,gate0), cin (input_2)
            '2,0': [0,  1,  1, 1],   # g5: c1 (layer0,gate1), c2 (layer1,gate1)
        },
        output_taps=[(1, 0), (2, 0)],   # sum, cout
    )

# ── Factory: 8-bit ripple-carry adder (16 inputs, 8 outputs, 40 gates) ───────

def make_adder8(name: str = 'adder8') -> GateNetwork:
    """
    8 ripple-carry full adder stages. Each stage is 3 layers of gates.
    Carry from stage N feeds into stage N+1 as cin.

    Inputs: [a0..a7, b0..b7]  (a_bits at indices 0-7, b_bits at 8-15)
    Outputs: [sum0..sum7]

    Topology per stage i (layers 3i, 3i+1, 3i+2):
      Layer 3i+0: g1(ai,bi)->s1   g2(ai,bi)->c1
      Layer 3i+1: g3(s1,cin)->sum  g4(s1,cin)->c2
      Layer 3i+2: g5(c1,c2)->cout_i   [cout_i becomes cin for stage i+1]

    Stage 0 cin = constant 0.0 (primary input index 16 = always 0).
    A synthetic 17th input (index 16) holds the value 0.0 as the initial carry.
    """
    n_inputs = 17  # 8 a-bits + 8 b-bits + 1 constant-0 for initial carry
    all_layers: list[list[SoftGate]] = []
    wiring: dict = {}
    output_taps: list[tuple] = []

    for stage in range(8):
        ai = stage       # primary input index for a-bit
        bi = stage + 8   # primary input index for b-bit

        # cin source: for stage 0, primary input 16 (constant 0).
        # For stage i>0, the carry output is at (3*(i-1)+2, 0).
        if stage == 0:
            cin_layer, cin_idx = -1, 16   # primary input
        else:
            cin_layer = 3 * (stage - 1) + 2
            cin_idx = 0

        base = 3 * stage

        # Layer base: g1(ai,bi)->s1  g2(ai,bi)->c1
        all_layers.append([
            SoftGate(f'{name}.s{stage}.g1(a{stage},b{stage})->s1'),
            SoftGate(f'{name}.s{stage}.g2(a{stage},b{stage})->c1'),
        ])
        wiring[f'{base},0'] = [-1, ai, -1, bi]
        wiring[f'{base},1'] = [-1, ai, -1, bi]

        # Layer base+1: g3(s1,cin)->sum  g4(s1,cin)->c2
        all_layers.append([
            SoftGate(f'{name}.s{stage}.g3(s1,cin)->sum'),
            SoftGate(f'{name}.s{stage}.g4(s1,cin)->c2'),
        ])
        wiring[f'{base+1},0'] = [base, 0, cin_layer, cin_idx]
        wiring[f'{base+1},1'] = [base, 0, cin_layer, cin_idx]

        # Layer base+2: g5(c1,c2)->cout
        all_layers.append([
            SoftGate(f'{name}.s{stage}.g5(c1,c2)->cout'),
        ])
        wiring[f'{base+2},0'] = [base, 1, base+1, 1]

        # Sum bit is the output of layer base+1, gate 0
        output_taps.append((base+1, 0))

    return GateNetwork(
        name=name,
        n_inputs=n_inputs,
        layers=all_layers,
        wiring=wiring,
        output_taps=output_taps,
    )

def adder8_examples() -> list:
    """Training examples for the 8-bit adder. Returns (inputs, targets) pairs."""
    examples = []
    for _ in range(400):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        result = (a + b) & 0xFF
        a_bits = [(a >> i) & 1 for i in range(8)]
        b_bits = [(b >> i) & 1 for i in range(8)]
        r_bits = [(result >> i) & 1 for i in range(8)]
        inputs  = a_bits + b_bits + [0]   # index 16 = constant 0 (carry-in)
        targets = r_bits
        examples.append((inputs, targets))
    return examples

def adder1_examples() -> list:
    """All 8 truth table entries for a 1-bit full adder."""
    return [
        ([a, b, cin], [(a+b+cin)&1, (a+b+cin)>>1])
        for a in range(2) for b in range(2) for cin in range(2)
    ]

# ── Training helper ───────────────────────────────────────────────────────────

def train(
    net: GateNetwork,
    examples: list,
    epochs: int = 6000,
    lr: float = 0.35,
    target_acc: float = 1.0,
    progress_cb=None,
) -> tuple:
    """
    Train net on examples for up to `epochs` steps.
    Returns (net, final_accuracy, epochs_run).
    progress_cb(epoch, acc) is called every 100 epochs if provided.
    """
    for ep in range(epochs):
        random.shuffle(examples)
        for inputs, targets in examples:
            net.train_step([float(x) for x in inputs], [float(t) for t in targets], lr)
        if ep % 100 == 0 or ep == epochs - 1:
            acc = net.accuracy(examples)
            if progress_cb:
                progress_cb(ep, acc)
            if acc >= target_acc:
                return net, acc, ep + 1
    return net, net.accuracy(examples), epochs
