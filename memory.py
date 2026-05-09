# memory.py — 256-byte memory backed entirely by a file on disk (no RAM)

import os

SIZE = 256

class Memory:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(bytes(SIZE))

    def read(self, addr):
        with open(self.path, 'rb') as f:
            f.seek(addr & 0xFF)
            return f.read(1)[0]

    def write(self, addr, value):
        with open(self.path, 'r+b') as f:
            f.seek(addr & 0xFF)
            f.write(bytes([value & 0xFF]))

    def load(self, data, start=0):
        for i, byte in enumerate(data):
            if start + i < SIZE:
                self.write(start + i, byte)

    def reset(self):
        with open(self.path, 'wb') as f:
            f.write(bytes(SIZE))

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
