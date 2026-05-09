# cpu.py — 8-bit CPU with fetch-decode-execute cycle
# All arithmetic goes through the gate-level ALU in alu.py.
# Pass turbo=True to CPU() to use native Python arithmetic instead —
# same behaviour, orders of magnitude faster, needed for real-time games.

from alu import (add8, sub8, and8, or8, xor8, not8, inc8, dec8,
                 shl8, shr8, mul8, int_to_bits)

# ── Turbo (native Python) ALU ─────────────────────────────────────────────────
def _t_add8(a, b, cin=0):
    r = (a + b + cin) & 0xFF; return r, 1 if (a + b + cin) > 0xFF else 0
def _t_sub8(a, b):
    r = (a - b) & 0xFF; return r, 1 if b > a else 0
def _t_and8(a, b): return (a & b) & 0xFF
def _t_or8 (a, b): return (a | b) & 0xFF
def _t_xor8(a, b): return (a ^ b) & 0xFF
def _t_not8(a):    return (~a) & 0xFF
def _t_inc8(a):    return _t_add8(a, 1)
def _t_dec8(a):    return _t_sub8(a, 1)
def _t_shl8(a):
    c = (a >> 7) & 1; return ((a << 1) & 0xFF), c
def _t_shr8(a):
    c = a & 1; return (a >> 1) & 0xFF, c
def _t_mul8(a, b):
    sa = a if a < 128 else a - 256
    sb = b if b < 128 else b - 256
    return max(-128, min(127, sa * sb)) & 0xFF

# ── Opcode table ─────────────────────────────────────────────────────────────
NOP      = 0x00
LDA_IMM  = 0x01;  LDA_ADDR = 0x02
STA_ADDR = 0x03
ADD_IMM  = 0x04;  ADD_ADDR = 0x05
SUB_IMM  = 0x06;  SUB_ADDR = 0x07
AND_IMM  = 0x08;  AND_ADDR = 0x09
OR_IMM   = 0x0A;  OR_ADDR  = 0x0B
XOR_IMM  = 0x0C;  XOR_ADDR = 0x0D
NOT_OP   = 0x0E
JMP_ADDR = 0x0F
JZ_ADDR  = 0x10;  JNZ_ADDR = 0x11
JC_ADDR  = 0x12;  JNC_ADDR = 0x13
JN_ADDR  = 0x14
INC_OP   = 0x15;  DEC_OP   = 0x16
OUT_OP   = 0x17;  OUTA_OP  = 0x18
INP_OP   = 0x19
LDX_IMM  = 0x1A;  LDX_ADDR = 0x1B;  STX_ADDR = 0x1C
TAX_OP   = 0x1D;  TXA_OP   = 0x1E
CMP_IMM  = 0x1F;  CMP_ADDR = 0x20
PUSH_OP  = 0x21;  POP_OP   = 0x22
JSR_ADDR = 0x23;  RET_OP   = 0x24
LDAX_OP  = 0x25;  STAX_OP  = 0x26   # indirect via X
SHL_OP   = 0x27;  SHR_OP   = 0x28
OUTN_OP  = 0x29                      # print A as decimal number
DRAW_OP  = 0x2A                      # render pong display from memory
KEY_OP   = 0x2B                      # non-blocking keyread → A (0 if none)
WAIT_OP  = 0x2C                      # sleep one frame (~50ms)
MUL_IMM  = 0x2D;  MUL_ADDR = 0x2E   # signed 8-bit multiply
# Long (16-bit address) jump variants — opcode + hi_byte + lo_byte
JMPL  = 0x30
JZL   = 0x31;  JNZL  = 0x32
JCL   = 0x33;  JNCL  = 0x34
JNL   = 0x35
HLT      = 0xFF

STACK_TOP = 0xEF   # stack lives at 0xE0–0xEF, SP starts here and grows down

class CPU:
    # Pong display constants (must match pong.asm memory layout)
    _PONG_BALL_X  = 0xF0
    _PONG_BALL_Y  = 0xF1
    _PONG_P1_Y    = 0xF4
    _PONG_P2_Y    = 0xF5
    _PONG_SCORE1  = 0xF6
    _PONG_SCORE2  = 0xF7
    _CW, _CH      = 40, 18   # court width / game-area height
    _PH           = 5        # paddle height

    def __init__(self, memory, verbose=False, turbo=False):
        self.mem          = memory
        self.A            = 0
        self.X            = 0
        self.PC           = 0
        self.SP           = STACK_TOP
        self.Z            = 0    # zero flag
        self.C            = 0    # carry flag
        self.N            = 0    # negative flag
        self.halted       = False
        self.verbose      = verbose
        self.turbo        = turbo   # True = native Python arithmetic (fast)
        self.cycles       = 0
        self._display_up  = False
        self.code         = bytearray(65536)
        self.code_end     = 0

        # Select ALU backend at init so step() has no per-op branch
        if turbo:
            self._add8 = _t_add8; self._sub8 = _t_sub8
            self._and8 = _t_and8; self._or8  = _t_or8
            self._xor8 = _t_xor8; self._not8 = _t_not8
            self._inc8 = _t_inc8; self._dec8 = _t_dec8
            self._shl8 = _t_shl8; self._shr8 = _t_shr8
            self._mul8 = _t_mul8
        else:
            self._add8 = add8;  self._sub8 = sub8
            self._and8 = and8;  self._or8  = or8
            self._xor8 = xor8;  self._not8 = not8
            self._inc8 = inc8;  self._dec8 = dec8
            self._shl8 = shl8;  self._shr8 = shr8
            self._mul8 = mul8

    # ── helpers ──────────────────────────────────────────────────────────────

    def _flags(self, result, carry=0):
        result &= 0xFF
        self.Z = 1 if result == 0 else 0
        self.C = carry & 1
        self.N = int_to_bits(result)[7]

    def load_code(self, bytecode, origin=0):
        """Load assembled bytecode into the code store."""
        for i, b in enumerate(bytecode):
            self.code[origin + i] = b
        self.code_end = origin + len(bytecode)

    def _fetch(self):
        byte = self.code[self.PC]
        self.PC += 1
        return byte

    def _fetch16(self):
        hi = self._fetch()
        lo = self._fetch()
        return (hi << 8) | lo

    def _push(self, val):
        self.mem.write(self.SP, val & 0xFF)
        self.SP = (self.SP - 1) & 0xFF

    def _pop(self):
        self.SP = (self.SP + 1) & 0xFF
        return self.mem.read(self.SP)

    # ── single step ──────────────────────────────────────────────────────────

    def step(self):
        if self.halted:
            return False
        op = self._fetch()
        self.cycles += 1

        if self.verbose:
            print(f"    [${self.PC-1:02X}] op=${op:02X}  "
                  f"A={self.A:02X} X={self.X:02X}  Z={self.Z} C={self.C} N={self.N}")

        if   op == NOP:      pass
        elif op == HLT:      self.halted = True; return False

        elif op == LDA_IMM:  self.A = self._fetch();                    self._flags(self.A)
        elif op == LDA_ADDR: self.A = self.mem.read(self._fetch());     self._flags(self.A)
        elif op == STA_ADDR: self.mem.write(self._fetch(), self.A)

        elif op == ADD_IMM:
            r, c = self._add8(self.A, self._fetch());  self.A = r; self._flags(r, c)
        elif op == ADD_ADDR:
            r, c = self._add8(self.A, self.mem.read(self._fetch())); self.A = r; self._flags(r, c)

        elif op == SUB_IMM:
            r, b = self._sub8(self.A, self._fetch());  self.A = r; self._flags(r, b)
        elif op == SUB_ADDR:
            r, b = self._sub8(self.A, self.mem.read(self._fetch())); self.A = r; self._flags(r, b)

        elif op == AND_IMM:  self.A = self._and8(self.A, self._fetch());      self._flags(self.A)
        elif op == AND_ADDR: self.A = self._and8(self.A, self.mem.read(self._fetch())); self._flags(self.A)
        elif op == OR_IMM:   self.A = self._or8(self.A, self._fetch());       self._flags(self.A)
        elif op == OR_ADDR:  self.A = self._or8(self.A, self.mem.read(self._fetch()));  self._flags(self.A)
        elif op == XOR_IMM:  self.A = self._xor8(self.A, self._fetch());      self._flags(self.A)
        elif op == XOR_ADDR: self.A = self._xor8(self.A, self.mem.read(self._fetch())); self._flags(self.A)
        elif op == NOT_OP:   self.A = self._not8(self.A);                     self._flags(self.A)

        elif op == JMP_ADDR: self.PC = self._fetch()
        elif op == JZ_ADDR:
            addr = self._fetch()
            if self.Z: self.PC = addr
        elif op == JNZ_ADDR:
            addr = self._fetch()
            if not self.Z: self.PC = addr
        elif op == JC_ADDR:
            addr = self._fetch()
            if self.C: self.PC = addr
        elif op == JNC_ADDR:
            addr = self._fetch()
            if not self.C: self.PC = addr
        elif op == JN_ADDR:
            addr = self._fetch()
            if self.N: self.PC = addr

        elif op == INC_OP:
            r, c = self._inc8(self.A); self.A = r; self._flags(r, c)
        elif op == DEC_OP:
            r, b = self._dec8(self.A); self.A = r; self._flags(r)

        elif op == OUT_OP:   print(chr(self.A), end='', flush=True)
        elif op == OUTN_OP:  print(self.A, end='', flush=True)

        elif op == DRAW_OP:  self._pong_draw()
        elif op == KEY_OP:   self.A = self._poll_key(); self._flags(self.A)
        elif op == WAIT_OP:
            import time; time.sleep(0.05)
        elif op == MUL_IMM:
            r = self._mul8(self.A, self._fetch()); self.A = r; self._flags(r)
        elif op == MUL_ADDR:
            r = self._mul8(self.A, self.mem.read(self._fetch())); self.A = r; self._flags(r)

        elif op == JMPL:  self.PC = self._fetch16()
        elif op == JZL:
            addr = self._fetch16()
            if self.Z: self.PC = addr
        elif op == JNZL:
            addr = self._fetch16()
            if not self.Z: self.PC = addr
        elif op == JCL:
            addr = self._fetch16()
            if self.C: self.PC = addr
        elif op == JNCL:
            addr = self._fetch16()
            if not self.C: self.PC = addr
        elif op == JNL:
            addr = self._fetch16()
            if self.N: self.PC = addr
        elif op == OUTA_OP:  print(f"{self.A}", end='', flush=True)
        elif op == INP_OP:
            try:
                self.A = int(input("input> ")) & 0xFF
            except (ValueError, EOFError):
                self.A = 0
            self._flags(self.A)

        elif op == LDX_IMM:  self.X = self._fetch();                    self._flags(self.X)
        elif op == LDX_ADDR: self.X = self.mem.read(self._fetch());     self._flags(self.X)
        elif op == STX_ADDR: self.mem.write(self._fetch(), self.X)
        elif op == TAX_OP:   self.X = self.A;                           self._flags(self.X)
        elif op == TXA_OP:   self.A = self.X;                           self._flags(self.A)
        elif op == LDAX_OP:  self.A = self.mem.read(self.X);            self._flags(self.A)
        elif op == STAX_OP:  self.mem.write(self.X, self.A)

        elif op == CMP_IMM:
            r, b = sub8(self.A, self._fetch()); self._flags(r, b)
        elif op == CMP_ADDR:
            r, b = sub8(self.A, self.mem.read(self._fetch())); self._flags(r, b)

        elif op == PUSH_OP:  self._push(self.A)
        elif op == POP_OP:   self.A = self._pop(); self._flags(self.A)
        elif op == JSR_ADDR:
            addr = self._fetch(); self._push(self.PC); self.PC = addr
        elif op == RET_OP:   self.PC = self._pop()

        elif op == SHL_OP:
            r, c = self._shl8(self.A); self.A = r; self._flags(r, c)
        elif op == SHR_OP:
            r, c = self._shr8(self.A); self.A = r; self._flags(r, c)

        else:
            print(f"\n  [FAULT] Unknown opcode ${op:02X} at PC=${self.PC-1:02X}")
            self.halted = True
            return False

        return True

    def run(self, max_cycles=500_000):
        self.halted = False
        while not self.halted:
            if not self.step():
                break
            if self.cycles >= max_cycles:
                print(f"\n  [HALT] Cycle limit ({max_cycles}) reached")
                break

    def reset(self):
        self.A = self.X = self.PC = 0
        self.SP = STACK_TOP
        self.Z = self.C = self.N = 0
        self.halted = False
        self.cycles = 0
        # code store intentionally preserved across reset

    _FRAME_S = 1 / 30   # target 30 fps
    _last_draw = 0.0

    def _pong_draw(self):
        import sys, os, ctypes, time
        now = time.perf_counter()
        gap = self._FRAME_S - (now - CPU._last_draw)
        if gap > 0:
            time.sleep(gap)
        CPU._last_draw = time.perf_counter()

        if not self._display_up:
            try:
                k = ctypes.windll.kernel32
                h = k.GetStdHandle(-11)
                mode = ctypes.c_ulong()
                k.GetConsoleMode(h, ctypes.byref(mode))
                k.SetConsoleMode(h, mode.value | 4)
            except Exception:
                pass
            sys.stdout.write('\033[2J\033[H\033[?25l')
            sys.stdout.flush()
            self._display_up = True
            CPU._last_draw = time.perf_counter()

        m   = self.mem
        bx  = m.read(self._PONG_BALL_X)
        by  = m.read(self._PONG_BALL_Y)
        p1y = m.read(self._PONG_P1_Y)
        p2y = m.read(self._PONG_P2_Y)
        s1  = m.read(self._PONG_SCORE1)
        s2  = m.read(self._PONG_SCORE2)

        W, H, PH = self._CW, self._CH, self._PH
        score = f"  {s1} : {s2}  "
        pad   = (W - len(score)) // 2
        top   = '#' * pad + score + '#' * (W - pad - len(score))

        lines = [top]
        for y in range(H):
            row = []
            for x in range(W):
                if x == 0 or x == W - 1:
                    row.append('#')
                elif x == 1 and p1y <= y < p1y + PH:
                    row.append('|')
                elif x == W - 2 and p2y <= y < p2y + PH:
                    row.append('|')
                elif x == bx and y == by:
                    row.append('O')
                else:
                    row.append(' ')
            lines.append(''.join(row))
        lines.append('#' * W)
        lines.append('  W/S = P1 (left)    O/L = P2 (right)    Q = quit')

        sys.stdout.write('\033[H' + '\n'.join(lines) + '\n')
        sys.stdout.flush()

    def _poll_key(self):
        try:
            import msvcrt
            return ord(msvcrt.getch()) if msvcrt.kbhit() else 0
        except ImportError:
            import sys, select, tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                if select.select([sys.stdin], [], [], 0)[0]:
                    return ord(sys.stdin.read(1))
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            return 0

    def teardown_display(self):
        if self._display_up:
            import sys
            sys.stdout.write('\033[2J\033[H\033[?25h')  # clear, restore cursor
            sys.stdout.flush()
            self._display_up = False

    def show_regs(self):
        bits = lambda v: f"{v:08b}"
        print(f"\n  == Registers ==")
        print(f"  A  = ${self.A:02X}  ({self.A:3d})  [{bits(self.A)}]")
        print(f"  X  = ${self.X:02X}  ({self.X:3d})  [{bits(self.X)}]")
        print(f"  PC = ${self.PC:02X}  SP = ${self.SP:02X}")
        print(f"  Flags: Z={self.Z}  C={self.C}  N={self.N}")
        print(f"  Cycles: {self.cycles}")
