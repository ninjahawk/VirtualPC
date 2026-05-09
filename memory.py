# memory.py — 256-byte memory backed entirely by a file on disk (no RAM)
# The file handle is kept open between operations so we pay seek cost
# rather than open/close cost on every access — the data still lives on
# disk and every read/write goes directly to the file.

import os

SIZE = 256

class Memory:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(bytes(SIZE))
        self._fh = open(path, 'r+b')

    def read(self, addr):
        self._fh.seek(addr & 0xFF)
        return self._fh.read(1)[0]

    def write(self, addr, value):
        self._fh.seek(addr & 0xFF)
        self._fh.write(bytes([value & 0xFF]))
        self._fh.flush()

    def load(self, data, start=0):
        for i, byte in enumerate(data):
            if start + i < SIZE:
                self.write(start + i, byte)

    def reset(self):
        self._fh.seek(0)
        self._fh.write(bytes(SIZE))
        self._fh.flush()

    def dump(self, start=0, length=256):
        length = min(length, SIZE - start)
        print(f"\n  Memory [{start:02X}–{start+length-1:02X}]")
        print("       ", end="")
        for i in range(16):
            print(f" {i:02X}", end="")
        print()
        for row in range(0, length, 16):
            print(f"  {start+row:02X}:  ", end="")
            for col in range(16):
                addr = start + row + col
                if addr < start + length:
                    print(f" {self.read(addr):02X}", end="")
                else:
                    print("   ", end="")
            print()
