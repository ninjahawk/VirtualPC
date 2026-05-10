# nand_is_dead.flux -- The old world had NAND gates.
#
# NAND is fixed. Its truth table never changes.
# You build every other gate by wiring NANDs together.
# XOR needs 4 NANDs. A full adder needs 9.
# The circuit is designed by a human. The logic is prescribed.
#
# In Flux, there is one gate type. It starts with a random truth table.
# You show it what you want. It becomes what is needed.
# No wiring. No design. No instruction set.
# Just: describe the input-output relationship. Gradient descent does the rest.
#
# Run: python flux.py programs/nand_is_dead.flux

# NAND -- the classical universal gate, discovered by gradient descent
learn nand
0 0 -> 1
0 1 -> 1
1 0 -> 1
1 1 -> 0

# XOR -- famously requires 4 NANDs to wire up. One Flux gate, one training run.
learn xor
0 0 -> 0
0 1 -> 1
1 0 -> 1
1 1 -> 0

# XNOR -- inverse of XOR
learn xnor
0 0 -> 1
0 1 -> 0
1 0 -> 0
1 1 -> 1

# 1-bit full adder -- the carry-propagating building block of all integer arithmetic
learn adder adder1

# Verify
run nand  1 1
run xor   1 0
run xnor  1 1
run adder 1 1 0
run adder 1 1 1

list
