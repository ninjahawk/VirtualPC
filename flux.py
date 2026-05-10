"""
flux.py -- The Flux language interpreter.

Flux's unique angle: you don't program a computer. You teach it.

Every other language makes you describe HOW to compute something.
Flux makes you describe WHAT you want. The machine figures out the hardware.
"Compilation" is gradient descent. "Programs" are trained gate weights.
There is no instruction set, no bytecode, no opcodes. Just learned logic.

When you write a Flux program, you're not writing code.
You're writing a training specification. The CPU runs itself.

Commands:
  learn <name>              -- enter examples interactively, then train
  learn <name> adder1       -- train the built-in 1-bit full adder
  learn <name> adder8       -- train the built-in 8-bit ripple-carry adder
  run   <name> [inputs...]  -- evaluate a trained program
  show  <name>              -- display gate truth tables and sharpness
  list                      -- list all trained programs
  forget <name>             -- delete a program
  help                      -- show this reference

Example interaction:
  flux> learn xor
  Enter examples (end with blank line):
    0 0 -> 0
    0 1 -> 1
    1 0 -> 1
    1 1 -> 0

  Training 'xor'...  epoch 500  acc=100%
  Done. Gates: [XOR]  sharpness=0.97

  flux> run xor 1 0
  1
"""

import os
import json
import random
import time
from typing import Optional, Callable

from soft_gate import (
    GateNetwork, SoftGate,
    make_single_gate, make_full_adder, make_adder8,
    adder1_examples, adder8_examples,
    train, GATE_NAMES,
)

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.fluxstate')

# ── Errors ────────────────────────────────────────────────────────────────────

class FluxError(Exception):
    pass

# ── Flux interpreter ──────────────────────────────────────────────────────────

class FluxInterpreter:

    def __init__(self, state_dir: str = STATE_DIR):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _prog_path(self, name: str) -> str:
        return os.path.join(self.state_dir, f'{name}.json')

    def _index_path(self) -> str:
        return os.path.join(self.state_dir, 'index.json')

    def _load_index(self) -> dict:
        p = self._index_path()
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return {}

    def _save_index(self, index: dict) -> None:
        with open(self._index_path(), 'w') as f:
            json.dump(index, f, indent=2)

    def _register(self, name: str, net: GateNetwork, acc: float, epochs: int) -> None:
        index = self._load_index()
        index[name] = {
            'n_gates':   net.n_gates(),
            'n_inputs':  net.n_inputs,
            'n_outputs': len(net.output_taps),
            'sharpness': round(net.sharpness(), 3),
            'accuracy':  round(acc, 3),
            'gates_id':  net.identify_all(),
            'trained_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'epochs':    epochs,
        }
        self._save_index(index)

    def _unregister(self, name: str) -> None:
        index = self._load_index()
        index.pop(name, None)
        self._save_index(index)

    def list_programs(self) -> dict:
        return self._load_index()

    def load_net(self, name: str) -> GateNetwork:
        p = self._prog_path(name)
        if not os.path.exists(p):
            raise FluxError(f"No program '{name}'. Type 'list' to see what's trained.")
        return GateNetwork.load(p)

    def _save_net(self, name: str, net: GateNetwork, acc: float, epochs: int) -> None:
        net.save(self._prog_path(name))
        self._register(name, net, acc, epochs)

    # ── Commands ──────────────────────────────────────────────────────────────

    def cmd_learn(
        self,
        name: str,
        mode: str = 'interactive',
        examples: Optional[list] = None,
        epochs: int = 6000,
        lr: float = 0.35,
        seed: Optional[int] = None,
        progress_cb: Optional[Callable] = None,
        print_fn=print,
    ) -> GateNetwork:
        """
        Train a program. mode: 'interactive' | 'adder1' | 'adder8' | 'examples'
        """
        if seed is not None:
            random.seed(seed)

        if mode == 'adder1':
            net = make_full_adder(name)
            examples = adder1_examples()
            print_fn(f"Training '{name}' as 1-bit full adder (5 gates)...")

        elif mode == 'adder8':
            net = make_adder8(name)
            random.seed(0)  # reproducible init for 40-gate network
            examples = adder8_examples()
            print_fn(f"Training '{name}' as 8-bit ripple-carry adder (40 gates)...")
            print_fn("  This may take a moment...")

        elif mode == 'examples':
            if not examples:
                raise FluxError("No examples provided.")
            n_in  = len(examples[0][0])
            n_out = len(examples[0][1])
            if n_in == 2 and n_out == 1:
                net = make_single_gate(name)
            elif n_in == 3 and n_out == 2:
                net = make_full_adder(name)
            else:
                net = _make_generic(name, n_in, n_out)
            print_fn(f"Training '{name}' ({net.n_gates()} gates, "
                     f"{n_in} inputs -> {n_out} outputs)...")

        elif mode == 'interactive':
            print_fn("Enter examples (input bits -> output bits). End with blank line.")
            examples = _read_examples_interactive(print_fn)
            if not examples:
                raise FluxError("No examples entered.")
            n_in  = len(examples[0][0])
            n_out = len(examples[0][1])
            if n_in == 2 and n_out == 1:
                net = make_single_gate(name)
            elif n_in == 3 and n_out == 2:
                net = make_full_adder(name)
            else:
                net = _make_generic(name, n_in, n_out)
            print_fn(f"Training '{name}' ({net.n_gates()} gates)...")

        else:
            raise FluxError(f"Unknown mode: {mode}")

        net, acc, ep = train(net, examples, epochs=epochs, lr=lr, progress_cb=progress_cb)
        self._save_net(name, net, acc, ep)

        gates_str = ', '.join(net.identify_all())
        if len(gates_str) > 50:
            gates_str = gates_str[:47] + '...'
        print_fn(f"Done. '{name}' acc={acc:.0%}  sharpness={net.sharpness():.2f}  "
                 f"gates=[{gates_str}]")
        return net

    def cmd_run(self, name: str, inputs: list) -> str:
        net = self.load_net(name)
        if len(inputs) != net.n_inputs:
            # The adder8 has 17 inputs (16 data + 1 carry-in constant)
            if net.n_inputs == 17 and len(inputs) == 16:
                inputs = list(inputs) + [0]
            elif net.n_inputs == 17 and len(inputs) == 2:
                # Interpret as decimal numbers for the 8-bit adder
                a, b = int(inputs[0]), int(inputs[1])
                a_bits = [(a >> i) & 1 for i in range(8)]
                b_bits = [(b >> i) & 1 for i in range(8)]
                inputs = a_bits + b_bits + [0]
            else:
                raise FluxError(
                    f"'{name}' expects {net.n_inputs} inputs, got {len(inputs)}."
                )
        float_inputs = [float(x) for x in inputs]
        outputs, _ = net.forward(float_inputs)
        rounded = [round(o) for o in outputs]

        n_out = len(net.output_taps)
        if n_out == 1:
            return str(rounded[0])
        elif n_out == 2:
            return f"out0={rounded[0]}  out1={rounded[1]}"
        elif n_out == 8:
            # 8-bit adder: decode to decimal
            val = sum(rounded[i] << i for i in range(8))
            return f"{val}  (bits: {''.join(str(rounded[i]) for i in range(7,-1,-1))})"
        else:
            return '  '.join(str(r) for r in rounded)

    def cmd_show(self, name: str) -> str:
        net = self.load_net(name)
        lines = [
            f"Program: {name}  ({net.n_gates()} gates, "
            f"{net.n_inputs} inputs, {len(net.output_taps)} outputs)",
            f"  Accuracy: see 'run' to test.  Sharpness: {net.sharpness():.2f}",
            '',
        ]
        for li, layer in enumerate(net.layers):
            for gi, gate in enumerate(layer):
                tt = gate.truth_table()
                fn = gate.identify()
                sharp = gate.sharpness()
                bar = '#' * round(sharp * 8) + '.' * (8 - round(sharp * 8))
                lines.append(
                    f"  L{li}G{gi}  {gate.label[:28]:28s}  {fn:8s}  "
                    f"[{bar}] {sharp:.2f}"
                )
                lines.append(
                    f"       f(0,0)={tt[0]:.2f}  f(0,1)={tt[1]:.2f}  "
                    f"f(1,0)={tt[2]:.2f}  f(1,1)={tt[3]:.2f}"
                )
        return '\n'.join(lines)

    def cmd_list(self) -> str:
        index = self._load_index()
        if not index:
            return "No programs trained yet. Use 'learn <name>' to train one."
        lines = [f"{'NAME':<14} {'GATES':>5}  {'IN':>3}  {'OUT':>3}  {'ACC':>5}  {'SHARP':>5}  FUNCTIONS"]
        lines.append('-' * 72)
        for name, meta in sorted(index.items()):
            fn_str = ', '.join(meta.get('gates_id', []))
            if len(fn_str) > 26:
                fn_str = fn_str[:23] + '...'
            lines.append(
                f"  {name:<12} {meta['n_gates']:>5}  {meta['n_inputs']:>3}  "
                f"{meta['n_outputs']:>3}  {meta['accuracy']:>4.0%}  "
                f"{meta['sharpness']:>5.2f}  {fn_str}"
            )
        return '\n'.join(lines)

    def cmd_forget(self, name: str) -> str:
        p = self._prog_path(name)
        if not os.path.exists(p):
            raise FluxError(f"No program '{name}'.")
        os.remove(p)
        self._unregister(name)
        return f"Forgot '{name}'."

    def cmd_help(self) -> str:
        return FLUX_HELP

    # ── Command dispatch ──────────────────────────────────────────────────────

    def execute(
        self,
        line: str,
        print_fn=print,
        progress_cb: Optional[Callable] = None,
    ) -> str:
        """Parse and execute one Flux line. Returns output string."""
        parts = line.strip().split()
        if not parts:
            return ''
        cmd = parts[0].lower()

        try:
            if cmd == 'learn':
                if len(parts) < 2:
                    raise FluxError("Usage: learn <name> [adder1|adder8]")
                name = parts[1]
                if len(parts) >= 3:
                    mode = parts[2].lower()
                    if mode not in ('adder1', 'adder8'):
                        raise FluxError(f"Unknown builtin '{mode}'. Try 'adder1' or 'adder8'.")
                    self.cmd_learn(name, mode=mode, print_fn=print_fn, progress_cb=progress_cb)
                else:
                    self.cmd_learn(name, mode='interactive', print_fn=print_fn, progress_cb=progress_cb)
                return ''

            elif cmd == 'run':
                if len(parts) < 2:
                    raise FluxError("Usage: run <name> [inputs...]")
                name = parts[1]
                inputs = [int(x) for x in parts[2:]]
                return self.cmd_run(name, inputs)

            elif cmd == 'show':
                if len(parts) < 2:
                    raise FluxError("Usage: show <name>")
                return self.cmd_show(parts[1])

            elif cmd == 'list':
                return self.cmd_list()

            elif cmd == 'forget':
                if len(parts) < 2:
                    raise FluxError("Usage: forget <name>")
                return self.cmd_forget(parts[1])

            elif cmd in ('help', '?'):
                return self.cmd_help()

            elif cmd in ('exit', 'quit', 'q'):
                return '__EXIT__'

            else:
                raise FluxError(f"Unknown command '{cmd}'. Type 'help' for commands.")

        except FluxError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: bad input -- {e}"

    def repl(self, prompt: str = 'flux> ') -> None:
        """Standalone interactive REPL."""
        print(FLUX_BANNER)
        while True:
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            result = self.execute(line)
            if result == '__EXIT__':
                break
            if result:
                print(result)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_examples_interactive(print_fn=print) -> list:
    """Read example lines from stdin until blank line."""
    examples = []
    while True:
        try:
            raw = input('  ')
        except EOFError:
            break
        raw = raw.strip()
        if not raw:
            break
        try:
            inp, out = _parse_example_line(raw)
            examples.append((inp, out))
            print_fn(f"    ({inp}) -> ({out})")
        except FluxError as e:
            print_fn(f"    Bad example: {e}  (format: 0 1 -> 1)")
    return examples

def _parse_example_line(line: str) -> tuple:
    """Parse '0 1 -> 1 0' into ([0,1], [1,0])."""
    if '->' not in line:
        raise FluxError("Missing '->' separator")
    left, right = line.split('->', 1)
    try:
        inp = [int(x) for x in left.split()]
        out = [int(x) for x in right.split()]
    except ValueError:
        raise FluxError("All values must be 0 or 1")
    if not inp or not out:
        raise FluxError("Empty input or output")
    if any(x not in (0, 1) for x in inp + out):
        raise FluxError("All values must be 0 or 1")
    return inp, out

def _make_generic(name: str, n_in: int, n_out: int) -> GateNetwork:
    """
    2-layer generic network for arbitrary I/O sizes.
    Layer 0: n_in gates (each reads two adjacent inputs, wrapping)
    Layer 1: n_out gates (each reads two adjacent layer-0 outputs, wrapping)
    """
    layers = [
        [SoftGate(f'{name}.h{i}') for i in range(n_in)],
        [SoftGate(f'{name}.o{i}') for i in range(n_out)],
    ]
    wiring = {}
    for i in range(n_in):
        wiring[f'0,{i}'] = [-1, i % n_in, -1, (i+1) % n_in]
    for i in range(n_out):
        wiring[f'1,{i}'] = [0, i % n_in, 0, (i+1) % n_in]
    output_taps = [(1, i) for i in range(n_out)]
    return GateNetwork(name, n_inputs=n_in, layers=layers,
                       wiring=wiring, output_taps=output_taps)


# ── Strings ───────────────────────────────────────────────────────────────────

FLUX_BANNER = """
  FLUX -- the language where you don't write code.
  You describe what you want. The hardware teaches itself.
  Type 'help' for commands.
"""

FLUX_HELP = """
  Flux command reference
  ----------------------
  learn <name>           enter examples interactively, then train
  learn <name> adder1    train a 1-bit full adder  (3 in, 2 out, 5 gates)
  learn <name> adder8    train an 8-bit adder       (16 in, 8 out, 40 gates)
  run   <name> [args]    evaluate a trained program on bit inputs
  show  <name>           display truth tables, sharpness, identified functions
  list                   list all trained programs
  forget <name>          delete a program
  help                   show this message
  exit                   quit

  For adder8, you can pass two decimal numbers:
    run <name> 10 20  ->  30

  Flux programs persist between runs in .fluxstate/
  No explicit save needed -- training writes the weights immediately.
"""


# ── Script execution ──────────────────────────────────────────────────────────

def run_script(path: str, interp: Optional['FluxInterpreter'] = None) -> None:
    """
    Execute a .flux script file.

    Script syntax:
      # comments are ignored
      learn <name>          -- next non-command lines are training examples
      0 1 -> 1              -- example line (inside a learn block)
      run <name> [args]     -- prints result
      show / list / forget  -- as in REPL

    Example .flux file:
      learn xor
      0 0 -> 0
      0 1 -> 1
      1 0 -> 1
      1 1 -> 0
      run xor 1 0
    """
    if interp is None:
        interp = FluxInterpreter()

    CMDS = {'learn', 'run', 'show', 'list', 'forget', 'help', 'exit', 'quit'}

    with open(path) as f:
        raw_lines = f.readlines()

    # Group into blocks: each 'learn' command absorbs following example lines.
    blocks: list[tuple[str, list[str]]] = []   # (command_line, [example_lines])
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].rstrip()
        stripped = line.strip()
        i += 1
        if not stripped or stripped.startswith('#'):
            continue
        first_word = stripped.split()[0].lower()
        if first_word == 'learn':
            examples: list[str] = []
            while i < len(raw_lines):
                el = raw_lines[i].strip()
                i += 1
                if not el or el.startswith('#'):
                    continue
                if el.split()[0].lower() in CMDS:
                    i -= 1
                    break
                examples.append(el)
            blocks.append((stripped, examples))
        else:
            blocks.append((stripped, []))

    for cmd_line, ex_lines in blocks:
        parts = cmd_line.split()
        if not parts:
            continue
        cmd = parts[0].lower()
        print(f'flux> {cmd_line}')

        if cmd == 'learn' and ex_lines:
            name = parts[1] if len(parts) > 1 else 'unnamed'
            if len(parts) >= 3 and parts[2].lower() in ('adder1', 'adder8'):
                result = interp.execute(cmd_line)
            else:
                parsed = []
                for el in ex_lines:
                    try:
                        inp, out = _parse_example_line(el)
                        parsed.append((inp, out))
                    except FluxError as e:
                        print(f'  bad example: {el!r}  ({e})')
                if parsed:
                    interp.cmd_learn(name, mode='examples', examples=parsed)
                result = None
        else:
            result = interp.execute(cmd_line)

        if result and result != '__EXIT__':
            print(result)
        if result == '__EXIT__':
            break


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1].endswith('.flux'):
        run_script(_sys.argv[1])
    else:
        FluxInterpreter().repl()
