# adder.flux -- Teach the machine binary addition.
#
# The full adder topology (5 gates) is fixed like a circuit board.
# The logic inside each gate is NOT fixed -- gradient descent writes it.
# After training, each gate discovers XOR, AND, or OR on its own.
#
# Run: python flux.py programs/adder.flux

# 1-bit full adder: (a, b, carry_in) -> (sum, carry_out)
learn adder adder1

# Verify all 8 truth table entries
run adder 0 0 0
run adder 0 1 0
run adder 1 0 0
run adder 1 1 0
run adder 1 1 1

# Inspect what logic the gates discovered
show adder
