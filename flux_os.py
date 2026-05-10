"""
flux_os.py -- FluxOS: graphical terminal OS for the Flux language.

An OS where every "program" is a trained gate network.
No instruction set. No opcodes. No fixed hardware.
The CPU is whatever gradient descent decided it should be.

The wallpaper is a live circuit diagram that updates as programs train.
Programs self-refine in the background (pass --refine to enable).
Weights persist to .fluxstate/ between runs.

Run: python flux_os.py
     python flux_os.py --refine    (background self-improvement)
"""

import os
import sys
import ctypes
import time
import random
import threading
from typing import Optional

from flux import FluxInterpreter, FluxError, FLUX_HELP
from soft_gate import GateNetwork, GATE_NAMES, adder1_examples, train

# ── ANSI ─────────────────────────────────────────────────────────────────────

class C:
    RST  = '\033[0m'
    BOLD = '\033[1m'
    DIM  = '\033[2m'
    BLK  = '\033[30m'
    RED  = '\033[31m'
    GRN  = '\033[32m'
    YLW  = '\033[33m'
    BLU  = '\033[34m'
    MGT  = '\033[35m'
    CYN  = '\033[36m'
    WHT  = '\033[37m'
    BGRN = '\033[92m'
    BYWL = '\033[93m'
    BBLU = '\033[94m'
    BMGT = '\033[95m'
    BCYN = '\033[96m'
    BWHT = '\033[97m'

def at(row, col): return f'\033[{row};{col}H'
def clr():        return '\033[2J\033[H'
def clrln():      return '\033[2K'
def hide():       return '\033[?25l'
def show():       return '\033[?25h'

def _enable_ansi():
    try:
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        m = ctypes.c_ulong()
        k.GetConsoleMode(h, ctypes.byref(m))
        k.SetConsoleMode(h, m.value | 4)
    except Exception:
        pass

def _termsize():
    try:
        s = os.get_terminal_size()
        return s.columns, s.lines
    except OSError:
        return 80, 24

# ── Wallpaper ─────────────────────────────────────────────────────────────────

# Maps identified function name -> (symbol, color)
_GSYM = {
    'AND':   ('[&]', C.GRN),
    'OR':    ('[|]', C.BLU),
    'XOR':   ('[^]', C.CYN),
    'NAND':  ('[!]', C.RED),
    'NOR':   ('[v]', C.MGT),
    'XNOR':  ('[=]', C.YLW),
    'NOT_A': ('[~]', C.WHT),
    'NOT_B': ('[~]', C.WHT),
    'A':     ('[A]', C.DIM),
    'B':     ('[B]', C.DIM),
    'TRUE':  ('[1]', C.BWHT),
    'FALSE': ('[0]', C.DIM),
}

def _gsym(name: str) -> tuple:
    return _GSYM.get(name, ('[?]', C.DIM))

def _wallpaper(cols: int, loaded_programs: dict) -> list[str]:
    """
    Render the FluxOS wallpaper: title + live circuit art.
    The circuit shows gate symbols colored by identified function.
    Unknown/unloaded slots are dim [?].
    Returns a list of strings (one per line, already ANSI-colored).
    """
    W = max(cols, 40)

    # Build a flat list of gate symbols from loaded programs
    all_gates: list[tuple[str, str]] = []  # (symbol, color)
    for name, net in loaded_programs.items():
        for g in [g for layer in net.layers for g in layer]:
            sym, col = _gsym(g.identify())
            all_gates.append((sym, col))
    # Pad with unknowns to fill the circuit row
    while len(all_gates) < (W // 5):
        all_gates.append(('[?]', C.DIM))

    # Line 1: title bar
    title = ' F L U X   O S '
    sub   = ' the computer that learns itself '
    pad_t = max(0, W - len(title) - len(sub) - 2)
    l1 = (f'{C.BOLD}{C.BWHT} {title}{C.RST}'
          f'{C.DIM}{C.WHT}{sub}{" "*pad_t}{C.RST}')

    # Line 2: circuit row 1 -- wires and gates
    def _circuit_row(gates, offset, w):
        out = f'{C.DIM}{C.CYN}'
        col = 0
        for i, (sym, color) in enumerate(gates[offset:offset + w//5]):
            out += f'---{color}{sym}{C.RST}{C.DIM}{C.CYN}'
            col += 5
            if col >= w - 5:
                break
        out += '---' + C.RST
        return out

    def _wire_row(gates, offset, w):
        # Vertical wires between alternating gates
        out = ''
        col = 0
        for i, (sym, color) in enumerate(gates[offset:offset + w//5]):
            gap = '   ' if i % 2 == 0 else '   '
            wire = '|' if i % 3 == 0 else ' '
            out += f'  {color}{wire}{C.RST}  '
            col += 5
            if col >= w - 5:
                break
        return out

    n = len(all_gates)
    l2 = _circuit_row(all_gates, 0,            W)
    l3 = _wire_row   (all_gates, 0,            W)
    l4 = _circuit_row(all_gates, min(W//5, n), W)
    l5 = f'{C.DIM}{C.CYN}{"─"*(W)}{C.RST}'

    return [l1, l2, l3, l4, l5]

# ── Program panel ─────────────────────────────────────────────────────────────

def _prog_panel(index: dict, height: int) -> list[str]:
    """Left panel: list of programs with sharpness bar."""
    lines = [f'{C.BOLD}{C.YLW}PROGRAMS{C.RST}', f'{C.DIM}{"─"*20}{C.RST}']
    for name, meta in sorted(index.items()):
        sharp = meta.get('sharpness', 0.0)
        acc   = meta.get('accuracy',  0.0)
        filled = round(sharp * 6)
        bar = f'{C.GRN}' + '#' * filled + f'{C.DIM}' + '.' * (6-filled) + C.RST
        fn_list = meta.get('gates_id', [])
        fn_str = fn_list[0] if fn_list else '?'
        acc_str = f'{acc:.0%}'
        nm = (name[:9] + '..') if len(name) > 11 else name
        lines.append(f'  {C.CYN}{nm:<11}{C.RST} {bar} {C.DIM}{acc_str}{C.RST}')
    while len(lines) < height:
        lines.append('')
    return lines[:height]

# ── Terminal scrollback ───────────────────────────────────────────────────────

class Terminal:
    def __init__(self, max_lines: int = 500):
        self._buf: list[str] = []
        self._max = max_lines

    def write(self, text: str):
        for line in str(text).splitlines():
            self._buf.append(line)
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def last(self, n: int) -> list[str]:
        lines = self._buf[-n:]
        while len(lines) < n:
            lines.insert(0, '')
        return lines

# ── Self-refining background thread ──────────────────────────────────────────

class Refiner:
    """
    Background thread that continuously refines trained programs.
    This is the "language that rewrites itself" -- programs keep
    gradient-descending even while you're doing other things.
    """

    def __init__(self, interp: FluxInterpreter, terminal: Terminal):
        self._interp  = interp
        self._term    = terminal
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status  = ''

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def status(self) -> str:
        return self._status

    def _loop(self):
        while self._running:
            index = self._interp.list_programs()
            for name, meta in index.items():
                if not self._running:
                    return
                sharp = meta.get('sharpness', 1.0)
                acc   = meta.get('accuracy',  1.0)
                if acc < 1.0 or sharp < 0.90:
                    self._refine_one(name, meta)
            time.sleep(5)

    def _refine_one(self, name: str, meta: dict):
        try:
            net = self._interp.load_net(name)
            n_in  = meta['n_inputs']
            n_out = meta['n_outputs']
            # Rebuild examples based on program type
            if n_in == 3 and n_out == 2:
                examples = adder1_examples()
            elif n_in == 17 and n_out == 8:
                examples = self._make_adder8_examples()
            else:
                return  # can't auto-generate examples for arbitrary programs

            self._status = f'refining {name}...'
            old_acc = net.accuracy(examples)
            net, acc, ep = train(net, examples, epochs=500, lr=0.2)
            if acc > old_acc + 0.01 or (acc == old_acc and net.sharpness() > meta['sharpness'] + 0.01):
                self._interp._save_net(name, net, acc, ep)
                self._term.write(
                    f'{C.DIM}[refiner] {name}: acc {old_acc:.0%} -> {acc:.0%}  '
                    f'sharp={net.sharpness():.2f}{C.RST}'
                )
            self._status = ''
        except Exception:
            self._status = ''

    def _make_adder8_examples(self):
        from soft_gate import adder8_examples
        return adder8_examples()

# ── FluxOS ────────────────────────────────────────────────────────────────────

PANEL_W  = 22    # left panel width
WALL_H   = 5     # wallpaper rows
STATUS_H = 1     # status bar rows

class FluxOS:

    def __init__(self, refine: bool = False):
        self._interp  = FluxInterpreter()
        self._term    = Terminal()
        self._refiner = Refiner(self._interp, self._term) if refine else None
        self._status  = 'ready'
        self._loaded: dict[str, GateNetwork] = {}  # name -> GateNetwork for wallpaper

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _loaded_nets(self) -> dict:
        """Load all programs for wallpaper rendering (cached)."""
        index = self._interp.list_programs()
        for name in index:
            if name not in self._loaded:
                try:
                    self._loaded[name] = self._interp.load_net(name)
                except Exception:
                    pass
        for name in list(self._loaded):
            if name not in index:
                del self._loaded[name]
        return self._loaded

    def _redraw(self):
        cols, rows = _termsize()
        cols  = max(cols, 60)
        rows  = max(rows, 20)

        term_h  = rows - WALL_H - STATUS_H - 3
        term_w  = cols - PANEL_W - 3

        out = [clr(), hide()]

        # ── Wallpaper ────────────────────────────────────────────────────────
        for i, wline in enumerate(_wallpaper(cols, self._loaded_nets())):
            out.append(at(i+1, 1) + wline)

        # ── Separator ────────────────────────────────────────────────────────
        sep_row = WALL_H + 1
        out.append(at(sep_row, 1) + C.DIM + '+' + '-'*(PANEL_W-1) + '+' + '-'*(term_w) + '+' + C.RST)

        # ── Program panel ────────────────────────────────────────────────────
        index = self._interp.list_programs()
        panel = _prog_panel(index, term_h + 1)
        for i, pline in enumerate(panel):
            out.append(at(sep_row + 1 + i, 1) + pline)

        # ── Terminal scrollback ───────────────────────────────────────────────
        recent = self._term.last(term_h)
        for i, tline in enumerate(recent):
            # Truncate to fit panel width
            visible = tline[:term_w] if tline else ''
            out.append(at(sep_row + 1 + i, PANEL_W + 2) + C.DIM + '|' + C.RST + ' ' + visible)

        # ── Bottom separator ──────────────────────────────────────────────────
        bot_row = sep_row + term_h + 1
        out.append(at(bot_row, 1) + C.DIM + '+' + '-'*(PANEL_W-1) + '+' + '-'*(term_w) + '+' + C.RST)

        # ── Status bar ────────────────────────────────────────────────────────
        n_prg   = len(index)
        ref_str = (f'  {C.YLW}[refiner: {self._refiner.status or "watching"}]{C.RST}'
                   if self._refiner else '')
        status_str = (f'{C.BOLD}{C.CYN} FluxOS{C.RST}'
                      f'{C.DIM} | {n_prg} program{"s" if n_prg!=1 else ""}'
                      f' | .fluxstate/ | {self._status}{C.RST}'
                      f'{ref_str}')
        out.append(at(bot_row + 1, 1) + clrln() + status_str)

        # ── Input prompt row ──────────────────────────────────────────────────
        prompt_row = bot_row + 2
        out.append(at(prompt_row, 1) + clrln() + f'{C.BGRN}flux>{C.RST} ')

        sys.stdout.write(''.join(out))
        sys.stdout.flush()

    # ── Command handling ──────────────────────────────────────────────────────

    def _progress(self, epoch: int, acc: float):
        """Progress callback during training -- updates status bar."""
        self._status = f'training... epoch {epoch}  acc={acc:.0%}'
        cols, rows = _termsize()
        bot_row = WALL_H + (rows - WALL_H - STATUS_H - 3) + 2
        sys.stdout.write(
            at(bot_row + 1, 1) + clrln() +
            f'{C.BOLD}{C.CYN} FluxOS{C.RST}'
            f'{C.DIM} | {self._status}{C.RST}'
        )
        sys.stdout.flush()

    def _handle(self, line: str):
        """Execute a Flux command and write output to terminal."""
        self._term.write(f'{C.BGRN}flux>{C.RST} {line}')

        def pf(text):
            self._term.write(str(text))

        result = self._interp.execute(
            line, print_fn=pf, progress_cb=self._progress
        )

        if result == '__EXIT__':
            return False
        if result:
            self._term.write(result)

        # Reload networks for wallpaper after learning
        if line.strip().lower().startswith('learn'):
            self._loaded.clear()

        self._status = 'ready'
        return True

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        _enable_ansi()

        if self._refiner:
            self._refiner.start()

        self._term.write(OS_BANNER)
        if self._refiner:
            self._term.write(
                f'{C.DIM}Background refiner active -- programs self-improve '
                f'between commands.{C.RST}'
            )

        try:
            while True:
                self._redraw()
                _, rows = _termsize()
                prompt_row = WALL_H + (rows - WALL_H - STATUS_H - 3) + 4
                sys.stdout.write(at(prompt_row, 1) + clrln() + f'{C.BGRN}flux> {C.RST}')
                sys.stdout.flush()
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip().lower() in ('exit', 'quit', 'q'):
                    break
                if not self._handle(line):
                    break
        finally:
            if self._refiner:
                self._refiner.stop()
            sys.stdout.write(show() + clr())
            sys.stdout.flush()
            print('FluxOS halted. Weights persisted to .fluxstate/')


# ── Banner / strings ──────────────────────────────────────────────────────────

OS_BANNER = f"""
{C.BOLD}{C.CYN}  +-------------------------------------------+{C.RST}
{C.BOLD}{C.CYN}  |  F L U X   O S  v1.0                     |{C.RST}
{C.CYN}  |  The AI is the computer.                  |{C.RST}
{C.DIM}{C.CYN}  |  Programs are trained gate networks.      |{C.RST}
{C.DIM}{C.CYN}  |  Compilation = gradient descent.          |{C.RST}
{C.DIM}{C.CYN}  +-------------------------------------------+{C.RST}

  Type {C.YLW}help{C.RST} for Flux commands.
  Try: {C.YLW}learn xor{C.RST}  or  {C.YLW}learn add adder1{C.RST}"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    refine = '--refine' in sys.argv
    FluxOS(refine=refine).run()
