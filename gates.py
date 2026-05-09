# gates.py — all logic derived from NAND (NAND is functionally complete)

def nand(a, b):
    return 0 if (a and b) else 1

def not_gate(a):       return nand(a, a)
def and_gate(a, b):    return not_gate(nand(a, b))
def or_gate(a, b):     return nand(not_gate(a), not_gate(b))
def nor_gate(a, b):    return not_gate(or_gate(a, b))
def xor_gate(a, b):
    n = nand(a, b)
    return nand(nand(a, n), nand(n, b))
def xnor_gate(a, b):   return not_gate(xor_gate(a, b))
def mux(a, b, sel):    return or_gate(and_gate(a, not_gate(sel)), and_gate(b, sel))
