#!/usr/bin/env python3
# vm.py — VirtualPC REPL
#
# Run:  python vm.py
#       python vm.py programs/hello.asm   (run a program directly)

import os
import sys

STATE_DIR = os.path.join(os.path.dirname(__file__), ".vpc_state")
os.makedirs(STATE_DIR, exist_ok=True)
MEM_FILE  = os.path.join(STATE_DIR, "memory.bin")

from memory import Memory
from cpu    import CPU
from assembler import assemble

BANNER = """
  ==========================================
       VirtualPC  --  8-bit virtual CPU
    Compute: NAND gates -> ALU -> CPU
    Memory : file on disk  (memory.bin)
  ==========================================
  Type 'help' for commands.
"""

HELP = """
  Commands
  ────────────────────────────────────────────────
  run  <file.asm>        assemble & run
  asm                    type assembly inline (end with '.')
  load <file.bin>        load raw binary into memory
  save <file.bin>        dump memory to binary file

  dump [start] [len]     hex dump of memory
  regs                   show CPU registers
  peek <addr>            read one memory byte
  poke <addr> <val>      write one memory byte

  reset                  reset CPU (keep memory)
  wipe                   reset CPU + clear memory
  trace                  toggle instruction trace
  list                   list .asm files here

  help                   this message
  exit                   quit (machine state saved)
  ────────────────────────────────────────────────
  Addresses and values accept decimal, $hex, or %binary.
"""

def parse_addr(s):
    if s.startswith('$'):  return int(s[1:], 16)
    if s.startswith('%'):  return int(s[1:], 2)
    return int(s)

def run_file(path, memory, cpu):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return
    with open(path, 'r') as f:
        source = f.read()
    code, origin, labels, errors = assemble(source)
    if errors:
        print("  Assembly errors:")
        for e in errors: print(f"    {e}")
        return
    print(f"  Assembled {len(code)} bytes at origin ${origin:02X}")
    if labels:
        print("  Labels: " + "  ".join(f"{k}=${v:02X}" for k, v in labels.items()))
    cpu.load_code(code, origin)   # code → CPU code store (Harvard arch)
    cpu.reset(); cpu.PC = origin
    print()
    try:
        cpu.run()
    finally:
        cpu.teardown_display()
    print()
    cpu.show_regs()

def main():
    memory = Memory(MEM_FILE)
    cpu    = CPU(memory)

    # Direct run mode: python vm.py program.asm
    if len(sys.argv) > 1:
        run_file(sys.argv[1], memory, cpu)
        return

    print(BANNER)

    while True:
        try:
            line = input("VPC> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye! State saved to", MEM_FILE)
            break

        if not line:
            continue
        parts = line.split()
        cmd   = parts[0].lower()

        if cmd in ('exit', 'quit', 'q'):
            print("  Bye! State saved to", MEM_FILE)
            break

        elif cmd == 'help':
            print(HELP)

        elif cmd == 'run':
            if len(parts) < 2:
                print("  Usage: run <file.asm>")
            else:
                run_file(parts[1], memory, cpu)

        elif cmd == 'asm':
            print("  Enter assembly lines. Finish with a bare '.'")
            lines = []
            while True:
                try:
                    ln = input("  ... ")
                except (EOFError, KeyboardInterrupt):
                    break
                if ln.strip() == '.':
                    break
                lines.append(ln)
            source = '\n'.join(lines)
            code, origin, labels, errors = assemble(source)
            if errors:
                for e in errors: print(f"  Error: {e}")
            else:
                print(f"  Assembled {len(code)} bytes")
                cpu.load_code(code, origin)
                cpu.reset(); cpu.PC = origin
                print()
                try:
                    cpu.run()
                finally:
                    cpu.teardown_display()
                print()
                cpu.show_regs()

        elif cmd == 'load':
            if len(parts) < 2:
                print("  Usage: load <file.bin>")
            else:
                try:
                    with open(parts[1], 'rb') as f:
                        data = f.read()
                    memory.load(data, start=0)
                    cpu.reset()
                    print(f"  Loaded {len(data)} bytes")
                except FileNotFoundError:
                    print(f"  File not found: {parts[1]}")

        elif cmd == 'save':
            if len(parts) < 2:
                print("  Usage: save <file.bin>")
            else:
                with open(parts[1], 'wb') as f:
                    for i in range(256):
                        f.write(bytes([memory.read(i)]))
                print(f"  Saved 256 bytes to {parts[1]}")

        elif cmd == 'dump':
            start  = parse_addr(parts[1]) if len(parts) > 1 else 0
            length = int(parts[2])        if len(parts) > 2 else 256
            memory.dump(start, length)

        elif cmd == 'regs':
            cpu.show_regs()

        elif cmd == 'peek':
            if len(parts) < 2:
                print("  Usage: peek <addr>")
            else:
                addr = parse_addr(parts[1])
                v    = memory.read(addr)
                print(f"  mem[${addr:02X}] = ${v:02X}  ({v})  [{v:08b}]")

        elif cmd == 'poke':
            if len(parts) < 3:
                print("  Usage: poke <addr> <val>")
            else:
                addr = parse_addr(parts[1])
                val  = parse_addr(parts[2])
                memory.write(addr, val)
                print(f"  mem[${addr:02X}] ← ${val:02X}")

        elif cmd == 'reset':
            cpu.reset()
            print("  CPU reset. Memory intact.")

        elif cmd == 'wipe':
            cpu.reset(); memory.reset()
            print("  CPU and memory cleared.")

        elif cmd == 'trace':
            cpu.verbose = not cpu.verbose
            print(f"  Instruction trace: {'ON' if cpu.verbose else 'OFF'}")

        elif cmd == 'list':
            files = [f for f in os.listdir('.') if f.endswith('.asm')]
            if files:
                print("  " + "  ".join(files))
            else:
                print("  No .asm files in current directory.")

        else:
            print(f"  Unknown command '{cmd}'. Type 'help'.")

if __name__ == '__main__':
    main()
