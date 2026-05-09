"""
run_ai_pong.py - Persistent AI Pong with LIVE in-game evolutionary training.

How the live training works (1+1 evolutionary strategy):
  - Memory holds the AI weights at $D0-$DA.  The CPU reads them every frame.
  - We track a 'best' weight set (Python side) and a 'challenger' (in memory).
  - Each rally, one side scores.  The CPU writes the score to $F6 / $F7.
  - When the AI scores: challenger wins -> promote it to 'best'.  Mutate it
    again into a new challenger, write to memory, play the next rally.
  - When you score:    challenger lost -> revert memory back to 'best',
    mutate again into a new challenger, play next rally.
  - On exit, the latest 'best' is what stays in memory.bin.

So: the AI literally improves while you play.  Score against it, revert.
Get scored on, promote.

Training stats live in:
  $F9 = generation # (mutations tried, mod 256)
  $FA = AI rallies won this session
  $FB = your rallies won this session
"""
import os, sys, random, copy
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from memory    import Memory
from cpu       import CPU
from assembler import assemble
import trainer
import simulate

STATE_DIR = os.path.join(ROOT, '.vpc_state')
os.makedirs(STATE_DIR, exist_ok=True)
mem = Memory(os.path.join(STATE_DIR, 'memory.bin'))

WEIGHT_ADDR = 0xD0
N_WEIGHTS   = 11
SCORE_P1    = 0xF6   # you
SCORE_P2    = 0xF7   # AI
GEN_ADDR    = 0xF9
AI_WIN_ADDR = 0xFA
PL_WIN_ADDR = 0xFB

MUT_SIGMA   = 2      # int8 stddev — small so each step is incremental
MUT_COUNT   = 2      # weights perturbed per mutation

def _signed(b): return b if b < 128 else b - 256
def _u8(s):     return max(-128, min(127, s)) & 0xFF

def get_weights():
    return [mem.read(WEIGHT_ADDR + i) for i in range(N_WEIGHTS)]

def set_weights(ws):
    for i, w in enumerate(ws):
        mem.write(WEIGHT_ADDR + i, w & 0xFF)

def mutate(ws):
    new = list(ws)
    idxs = random.sample(range(N_WEIGHTS), MUT_COUNT)
    for i in idxs:
        s = _signed(new[i]) + int(round(random.gauss(0, MUT_SIGMA)))
        new[i] = _u8(s)
    return new

# ── Bootstrap weights (train once if memory.bin is empty) ─────────────────────
print("=" * 60)
print("  VirtualPC - AI Pong with LIVE evolutionary training")
print("  W/S = your paddle   Q = quit")
print("  AI is mutating between every rally; weights persist on quit.")
print("=" * 60)

if not trainer.weights_are_trained(mem):
    print("\nNo weights in memory.bin — seeding with a quick gradient pass...")
    random.seed(7)
    data = trainer.make_data(trainer.N_TRAIN)
    W1, b1, W2, b2 = trainer.train(data)
    weights = trainer.weights_to_bytes(W1, b1, W2, b2)
    trainer.save_weights(mem, weights)
    print("Seed weights saved.\n")

best = get_weights()
W = list(best)
hits, total, rate = simulate.evaluate(W)
print(f"\nStarting return rate (sim): {hits}/{total} ({rate:.1f}%) — {simulate.grade(rate)}")
trainer.print_weights(best)

# Reset training counters in memory for this session
mem.write(GEN_ADDR,    0)
mem.write(AI_WIN_ADDR, 0)
mem.write(PL_WIN_ADDR, 0)

# Install first challenger (a small mutation of the seed)
challenger = mutate(best)
set_weights(challenger)
gen = 1

# ── Assemble and run the game loop step-by-step so we can mutate live ─────────
with open(os.path.join(ROOT, 'programs', 'ai_pong.asm')) as f:
    src = f.read()
code, origin, labels, errors = assemble(src)
if errors:
    print("Assembly errors:", errors); sys.exit(1)

cpu = CPU(mem)
cpu.load_code(code, origin)
cpu.PC = origin

last_p1 = mem.read(SCORE_P1)
last_p2 = mem.read(SCORE_P2)
ai_wins = pl_wins = 0
history = []   # log of (gen, outcome) for end-of-session summary

print("\nStarting game in 1.5s...\n")
import time; time.sleep(1.5)

CHECK_EVERY = 200   # CPU steps between score polls — cheap and frequent enough

try:
    since_check = 0
    while not cpu.halted:
        cpu.step()
        since_check += 1
        if since_check < CHECK_EVERY:
            continue
        since_check = 0

        p1 = mem.read(SCORE_P1)
        p2 = mem.read(SCORE_P2)
        if p2 != last_p2 and ((p2 - last_p2) & 0xFF) > 0:
            # AI scored: challenger wins, promote it
            best = list(challenger)
            ai_wins += 1
            history.append((gen, 'kept'))
            challenger = mutate(best)
            set_weights(challenger)
            gen += 1
            mem.write(GEN_ADDR,    gen & 0xFF)
            mem.write(AI_WIN_ADDR, ai_wins & 0xFF)
        elif p1 != last_p1 and ((p1 - last_p1) & 0xFF) > 0:
            # Player scored: challenger lost, revert
            set_weights(best)
            challenger = mutate(best)
            set_weights(challenger)
            pl_wins += 1
            history.append((gen, 'reverted'))
            gen += 1
            mem.write(GEN_ADDR,    gen & 0xFF)
            mem.write(PL_WIN_ADDR, pl_wins & 0xFF)
        last_p1, last_p2 = p1, p2
finally:
    # Make sure the surviving 'best' is what's persisted, not the last challenger
    set_weights(best)
    cpu.teardown_display()

print()
print(f"Generations tried this session: {gen-1}")
print(f"  AI won (mutation kept):     {ai_wins}")
print(f"  You won (mutation reverted): {pl_wins}")
hits, total, rate = simulate.evaluate(get_weights())
print(f"  Final return rate (sim):    {hits}/{total} ({rate:.1f}%) — {simulate.grade(rate)}")
print(f"\nFinal score  You: {mem.read(SCORE_P1)}   AI: {mem.read(SCORE_P2)}")
