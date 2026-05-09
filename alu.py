# alu.py — Arithmetic Logic Unit built from logic gates

from gates import and_gate, or_gate, xor_gate, not_gate

# Precompute bit representations for all 256 byte values so int_to_bits
# never allocates a new list at runtime — the NAND gates still do all the
# actual computation, this just removes the Python list overhead around them.
_INT_TO_BITS = tuple(tuple((n >> i) & 1 for i in range(8)) for n in range(256))

def int_to_bits(n, width=8):
    """Integer → [LSB, ..., MSB]"""
    return _INT_TO_BITS[int(n) & 0xFF]

def bits_to_int(bits):
    """[LSB, ..., MSB] → integer"""
    return (bits[0] | bits[1]<<1 | bits[2]<<2 | bits[3]<<3 |
            bits[4]<<4 | bits[5]<<5 | bits[6]<<6 | bits[7]<<7)

def half_adder(a, b):
    return xor_gate(a, b), and_gate(a, b)          # sum, carry

def full_adder(a, b, cin):
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    return s2, or_gate(c1, c2)                     # sum, carry_out

def add8(a, b, cin=0):
    """8-bit ripple-carry adder via chained full adders."""
    ab, bb = int_to_bits(a), int_to_bits(b)
    result, carry = [], cin
    for i in range(8):
        s, carry = full_adder(ab[i], bb[i], carry)
        result.append(s)
    return bits_to_int(result), carry

def sub8(a, b):
    """Subtraction via two's complement: a - b = a + NOT(b) + 1"""
    not_b = bits_to_int([not_gate(x) for x in int_to_bits(b)])
    result, carry = add8(a, not_b, cin=1)
    borrow = not_gate(carry)                        # borrow = NOT carry
    return result, borrow

def and8(a, b):
    ab, bb = int_to_bits(a), int_to_bits(b)
    return bits_to_int([and_gate(ab[i], bb[i]) for i in range(8)])

def or8(a, b):
    ab, bb = int_to_bits(a), int_to_bits(b)
    return bits_to_int([or_gate(ab[i], bb[i]) for i in range(8)])

def xor8(a, b):
    ab, bb = int_to_bits(a), int_to_bits(b)
    return bits_to_int([xor_gate(ab[i], bb[i]) for i in range(8)])

def not8(a):
    return bits_to_int([not_gate(x) for x in int_to_bits(a)])

def inc8(a):  return add8(a, 1)
def dec8(a):  return sub8(a, 1)

def shl8(a):
    bits = int_to_bits(a)
    carry = bits[7]
    return bits_to_int((0,) + bits[:7]), carry      # shift left, LSB=0

def shr8(a):
    bits = int_to_bits(a)
    carry = bits[0]
    return bits_to_int(bits[1:] + (0,)), carry      # shift right, MSB=0

def mul8(a, b):
    """Signed 8-bit × 8-bit → signed 8-bit (saturated).
    Implemented as shift-and-add on unsigned magnitudes, gate-level."""
    # sign bits
    a_sign = (a >> 7) & 1
    b_sign = (b >> 7) & 1
    # magnitudes via two's complement negation when negative
    a_mag = (bits_to_int([not_gate(x) for x in int_to_bits(a)]) + 1) & 0xFF if a_sign else a
    b_mag = (bits_to_int([not_gate(x) for x in int_to_bits(b)]) + 1) & 0xFF if b_sign else b
    # shift-and-add (Russian peasant) on magnitudes using gate-level adder
    acc = 0
    addend = a_mag
    multiplier = b_mag
    for _ in range(8):
        if multiplier & 1:                          # test LSB
            acc, _ = add8(acc, addend)              # accumulate
        addend = (addend << 1) & 0xFF               # shift addend left
        multiplier >>= 1                            # shift multiplier right
    # apply sign: XOR of input signs
    result_neg = xor_gate(a_sign, b_sign)
    if result_neg:
        acc = (bits_to_int([not_gate(x) for x in int_to_bits(acc)]) + 1) & 0xFF
    # saturate to signed 8-bit
    acc_signed = acc if acc < 128 else acc - 256
    acc_signed = max(-128, min(127, acc_signed))
    return acc_signed & 0xFF
