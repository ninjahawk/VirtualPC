"""
run_ai_pong.py - Persistent neural net Pong.

First run:  trains the network, saves weights to memory.bin, plays.
Later runs: loads saved weights from memory.bin, plays immediately.

The AI brain lives in memory.bin at $D0-$DA.
Retrain anytime with:  python trainer.py
"""
import os, sys
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from memory    import Memory
from cpu       import CPU
from assembler import assemble
import trainer

STATE_DIR = os.path.join(ROOT, '.vpc_state')
os.makedirs(STATE_DIR, exist_ok=True)
mem = Memory(os.path.join(STATE_DIR, 'memory.bin'))

print("=" * 52)
print("  VirtualPC - Persistent Neural Net Pong")
print("  Left paddle: W/S   Right paddle: AI   Q: quit")
print("=" * 52)

if trainer.weights_are_trained(mem):
    print("\nLoading saved weights from memory.bin...")
    weights = trainer.load_weights(mem)
    trainer.print_weights(weights)
    print("\n(AI was already trained - brain loaded from disk)")
else:
    print("\nNo saved weights found. Training from scratch...")
    print("(This only happens once)\n")
    random_state = __import__('random')
    random_state.seed(7)
    data = trainer.make_data(trainer.N_TRAIN)
    W1, b1, W2, b2 = trainer.train(data)
    weights = trainer.weights_to_bytes(W1, b1, W2, b2)
    trainer.save_weights(mem, weights)
    print("\nWeights saved to memory.bin - won't need to train again!")
    trainer.print_weights(weights)

# -- Assemble and run ----------------------------------------------------------
with open(os.path.join(ROOT, 'programs', 'ai_pong.asm')) as f:
    src = f.read()
code, origin, labels, errors = assemble(src)
if errors:
    print("Assembly errors:", errors); sys.exit(1)

cpu = CPU(mem)
cpu.load_code(code, origin)
cpu.PC = origin

print("\nStarting (W/S = you  |  right paddle = neural net  |  Q = quit)\n")
try:
    cpu.run(max_cycles=999_999_999)
finally:
    cpu.teardown_display()

print()
cpu.show_regs()
print(f"\nFinal score  You: {mem.read(0xF6)}   AI: {mem.read(0xF7)}")
