# logic.flux -- Learn basic boolean logic from examples.
#
# In Flux, you don't implement logic. You show examples.
# Gradient descent figures out the gate truth tables.
#
# Run: python flux.py programs/logic.flux

# XOR -- the gate that returns 1 when inputs differ
learn xor
0 0 -> 0
0 1 -> 1
1 0 -> 1
1 1 -> 0

# AND -- both inputs must be 1
learn and
0 0 -> 0
0 1 -> 0
1 0 -> 0
1 1 -> 1

# OR -- at least one input must be 1
learn or
0 0 -> 0
0 1 -> 1
1 0 -> 1
1 1 -> 1

# Verify all three
run xor 0 1
run xor 1 1
run and 1 1
run and 1 0
run or  0 0
run or  1 0

# Show what the gates discovered
show xor
list
