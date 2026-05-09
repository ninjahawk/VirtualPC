# VirtualPC

The idea: build a real 8-bit computer from the absolute bottom up, starting from a single NAND gate and stacking up through logic gates, an ALU, a CPU, an assembler, and a virtual machine REPL. Memory is backed by a file on disk. Programs are written in a custom assembly language. The AI layer trains a tiny neural network in Python and runs inference natively on the virtual CPU in assembly. You end up with a working machine that can run Pong, compute Fibonacci sequences, and host an AI opponent — all derived from `nand(a, b)`.

## How it works

The repo is deliberately kept small and only really has a handful of files that matter:

- **`gates.py`** — the absolute foundation. Every logic operation (NOT, AND, OR, XOR, MUX) is derived from a single NAND gate. Not modified.
- **`alu.py`** — the 8-bit Arithmetic Logic Unit built from those gates. Implements add, subtract, multiply, shift, and bitwise operations via ripple-carry adders and shift-and-add multipliers. Not modified.
- **`cpu.py`** — the CPU itself. Fetch-decode-execute cycle, registers (A, X, PC, SP), flags (Z, C, N), a full opcode table (~45 instructions), stack, subroutines, and special I/O ops for the pong display. **This file is where new instructions live.**
- **`assembler.py`** — two-pass assembler for the custom assembly language. Supports labels, immediates, `.org`/`.byte`/`.str` directives, and all addressing modes. **Write programs by targeting this.**
- **`vm.py`** — the REPL entry point. Run programs, inspect memory, poke registers, toggle trace mode. **This is what you run.**
- **`trainer.py`** — trains a 3→2→1 ReLU neural network to play Pong and saves quantized weights to `memory.bin`. **This file is edited and iterated on by the human.**
- **`run_ai_pong.py`** — loads saved weights (or trains fresh ones on first run) then executes `ai_pong.asm`, where inference runs natively on the virtual CPU in assembly.

By design, the entire machine fits in your head. The metric the AI opponent optimizes is simple: track the ball. Weights live in `memory.bin` at `$D0–$DA` and persist between runs without any explicit save step.

## Quick start

**Requirements:** Python 3.10+, no other dependencies.

```bash
# 1. Clone the repo
git clone https://github.com/ninjahawk/VirtualPC.git
cd VirtualPC

# 2. Run the virtual machine REPL
python vm.py

# 3. Or run a program directly
python vm.py programs/hello.asm
```

If the above commands all work ok, your setup is working and you can start writing assembly or playing Pong.

## Running the AI Pong

Simply run:

```bash
python run_ai_pong.py
```

On first run the neural net trains from scratch (takes about a second) and saves weights to `.vpc_state/memory.bin`. On every subsequent run the weights are loaded instantly and the AI plays immediately. The `trainer.py` file is essentially the human's side of this setup — point it at new hyperparameters and let it go.

Controls: `W`/`S` = your paddle (left) &nbsp;|&nbsp; right paddle = neural net &nbsp;|&nbsp; `Q` = quit.

## Project structure

```
gates.py        — NAND-complete logic gate library (foundation, do not modify)
alu.py          — 8-bit ALU built from gates (do not modify)
cpu.py          — CPU, opcode table, fetch-decode-execute cycle
assembler.py    — two-pass assembler for the custom assembly language
memory.py       — 256-byte file-backed memory (no RAM)
vm.py           — REPL entry point
trainer.py      — neural net trainer (human modifies this)
run_ai_pong.py  — persistent AI Pong runner
programs/       — example assembly programs
```

## Design choices

- **NAND-complete foundation.** Everything — addition, subtraction, multiplication, shifts — is ultimately derived from `nand(a, b)`. This is not a performance choice; it is a pedagogical one. The CPU is intentionally slow. The point is that every operation traces back to a single gate with no shortcuts taken anywhere in the stack.
- **File-backed memory.** The 256-byte address space is backed entirely by a file on disk (`memory.bin`). There is no in-process RAM. Machine state persists across runs without any explicit save step, and neural net weights written by the trainer are immediately visible to the CPU on next boot.
- **Harvard architecture.** Code lives in a separate store inside the CPU object; data lives in `memory.bin`. Programs of any length cannot corrupt their own data, and the two address spaces never collide regardless of program size.
- **Single-file assembler.** The two-pass assembler handles labels, all numeric bases (`$hex`, `%binary`, decimal), string literals, and `.org` directives in a single file with no dependencies beyond the opcode table in `cpu.py`. Writing a new program means writing a `.asm` file; no toolchain required.
- **Neural net inference in assembly.** The matrix-vector multiply, ReLU activations, and sign-of-output decision for the Pong AI run entirely in the custom assembly language on the virtual CPU. Weights are quantized to signed 8-bit integers and stored in `memory.bin`. The trainer is the only Python that touches the network; everything else is assembly running on a CPU built from NAND gates.

## Platform support

This code requires Python 3.10+ and runs on Windows, macOS, and Linux with no additional dependencies. The pong display uses ANSI escape codes; on Windows, ANSI support is enabled automatically via the Console API. Key input uses `msvcrt` on Windows and `termios`/`select` on POSIX — both paths are wired up in `cpu.py` and selected at runtime.

If you are running in an environment without a real terminal (e.g. some CI runners or headless IDEs), `DRAW`, `KEY`, and `WAIT` instructions will still execute but the display may not render correctly. All other instructions work unconditionally in any environment.

Seeing as the whole machine is pure Python with no native extensions, it runs fine on any hardware including low-end laptops and Raspberry Pis. It is just slow — a program that takes microseconds on real silicon may take milliseconds here. That is the point.

## License

MIT
