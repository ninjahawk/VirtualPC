"""
simulate.py - Headless pong simulation to evaluate the trained AI.

Serves the ball to the AI from every angle and height, counts returns.
Run standalone:  python simulate.py
Called automatically by run_ai_pong.py before each game.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from memory import Memory
from alu import mul8, add8, int_to_bits

STATE_DIR = os.path.join(os.path.dirname(__file__), '.vpc_state')

def _signed(b): return b if b < 128 else b - 256
def _relu8(v):  return max(0, _signed(v)) & 0xFF

def _predict_arrival(bx, by, dy):
    """Compute where ball will land at x=38. Matches assembly prediction."""
    frames = 38 - bx
    dy_s = 1 if dy == 1 else -1
    arrival = by + dy_s * frames
    if arrival < 0:    arrival = -arrival
    if arrival >= 17:  arrival = 34 - arrival
    return max(0, min(16, arrival))

def _ai_move(bx, by, p2y, dy, W):
    arrival = _predict_arrival(bx, by, dy)
    x0 = arrival >> 1
    x1 = (p2y + 2) >> 1
    x2 = 1 if dy == 1 else 0
    acc = mul8(x0, W[0]); t = mul8(x1, W[1]); acc, _ = add8(acc, t)
    acc, _ = add8(acc, mul8(x2, W[2])); acc, _ = add8(acc, W[3]); h1 = _relu8(acc)
    acc = mul8(x0, W[4]); t = mul8(x1, W[5]); acc, _ = add8(acc, t)
    acc, _ = add8(acc, mul8(x2, W[6])); acc, _ = add8(acc, W[7]); h2 = _relu8(acc)
    acc = mul8(h1, W[8]); t = mul8(h2, W[9]); acc, _ = add8(acc, t); acc, _ = add8(acc, W[10])
    n = int_to_bits(acc)[7]; z = (acc == 0)
    if not z:
        if n and p2y > 0:        p2y -= 1
        elif not n and p2y < 12: p2y += 1
    return p2y

def evaluate(W, verbose=False):
    """
    Serve the ball to the AI from every starting height (0-17) and
    both vertical directions.  Count how many times it returns.
    Returns (hits, total, return_rate_pct).
    """
    hits = total = 0
    misses_detail = []

    for start_by in range(18):
        for start_dy in (1, 255):          # 255 = -1 unsigned (going up)
            bx, by   = 19, start_by
            dx, dy   = 1,  start_dy        # serve toward AI
            p2y      = 6                   # AI starts at center

            for _ in range(250):           # max frames per serve
                p2y = _ai_move(bx, by, p2y, dy, W)

                # move ball
                bx += 1
                by_next = by + (1 if dy == 1 else -1)

                # bounce top / bottom (assembly bounces at exactly 0 and 17,
                # then adds new dy to the already-moved position)
                if by_next <= 0 or by_next >= 17:
                    dy = 255 if dy == 1 else 1
                    by_next = by_next + (1 if dy == 1 else -1)
                by = by_next

                # ball reached AI side
                if bx >= 38:
                    total += 1
                    if p2y <= by < p2y + 5:
                        hits += 1
                    else:
                        misses_detail.append((start_by, start_dy, by, p2y))
                    break

    rate = 100.0 * hits / total if total else 0.0

    if verbose and misses_detail:
        print("  Missed balls (start_by, start_dy, arrival_by, paddle_y):")
        for item in misses_detail[:10]:
            print(f"    {item}")

    return hits, total, rate

def grade(rate):
    if rate >= 90: return "excellent"
    if rate >= 75: return "good"
    if rate >= 55: return "okay"
    return "poor — consider retraining"

if __name__ == '__main__':
    mem_path = os.path.join(STATE_DIR, 'memory.bin')
    if not os.path.exists(mem_path):
        print("No memory.bin found — run trainer.py first.")
        sys.exit(1)
    mem = Memory(mem_path)
    W   = [mem.read(0xD0 + i) for i in range(11)]
    print("Evaluating AI against all 36 serve angles...")
    hits, total, rate = evaluate(W, verbose=True)
    print(f"\n  Return rate: {hits}/{total}  ({rate:.1f}%)  — {grade(rate)}")
