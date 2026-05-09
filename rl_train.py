"""
rl_train.py - Train the Pong AI through evolutionary self-play.

Starts from completely random weights and improves purely from game outcomes.
Each generation: simulate games with the current population, keep the best
performers, mutate them to create the next generation.  Watch the score climb.

Usage:
    python rl_train.py            # train from scratch, save weights, play
    python rl_train.py --notplay  # train and save only
"""
import random, copy, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from memory  import Memory
import simulate

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_POP    = 16     # individuals per generation
N_GEN    = 80     # generations to run
N_ELITE  = 4      # top survivors carried forward unchanged
SIGMA    = 0.35   # mutation noise (float weight space)
SCALE    = 4      # float → int8 quantisation factor (must match trainer.py)
SEED     = None   # set an int for reproducibility

STATE_DIR   = os.path.join(os.path.dirname(__file__), '.vpc_state')
WEIGHT_ADDR = 0xD0

# ── Weight encoding ───────────────────────────────────────────────────────────

def _q(v):
    return max(-128, min(127, int(round(v * SCALE)))) & 0xFF

def to_int8(ind):
    """Quantise a float individual to the 11-byte layout used by the CPU."""
    W1, b1, W2, b2 = ind['W1'], ind['b1'], ind['W2'], ind['b2']
    return [
        _q(W1[0][0]), _q(W1[1][0]), _q(W1[2][0]), _q(b1[0]),
        _q(W1[0][1]), _q(W1[1][1]), _q(W1[2][1]), _q(b1[1]),
        _q(W2[0]),    _q(W2[1]),    _q(b2),
    ]

def save(mem, ind):
    for i, w in enumerate(to_int8(ind)):
        mem.write(WEIGHT_ADDR + i, w)

# ── Population operations ─────────────────────────────────────────────────────

def random_individual():
    return {
        'W1': [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(3)],
        'b1': [random.gauss(0, 0.5) for _ in range(2)],
        'W2': [random.gauss(0, 1.0) for _ in range(2)],
        'b2': 0.0,
    }

def mutate(ind):
    n = copy.deepcopy(ind)
    for j in range(3):
        for i in range(2):
            n['W1'][j][i] += random.gauss(0, SIGMA)
    for i in range(2):
        n['b1'][i] += random.gauss(0, SIGMA)
        n['W2'][i] += random.gauss(0, SIGMA)
    n['b2']      += random.gauss(0, SIGMA * 0.3)
    return n

def fitness(ind):
    """Return rate 0–100 from the headless simulator."""
    _, _, rate = simulate.evaluate(to_int8(ind))
    return rate

# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    os.makedirs(STATE_DIR, exist_ok=True)
    if SEED is not None:
        random.seed(SEED)

    pop = [random_individual() for _ in range(N_POP)]

    best_ind  = None
    best_rate = -1.0

    W = 20   # progress-bar width
    header = f"{'Gen':>4}  {'Best%':>6}  {'Avg%':>6}  Progress"
    print(header)
    print("-" * (len(header) + 4))

    for gen in range(1, N_GEN + 1):
        scored = sorted([(fitness(ind), ind) for ind in pop], key=lambda x: -x[0])

        gen_best = scored[0][0]
        gen_avg  = sum(s for s, _ in scored) / len(scored)

        improved = ""
        if gen_best > best_rate:
            best_rate = gen_best
            best_ind  = scored[0][1]
            improved  = "  *"

        filled = int(gen_best / 100 * W)
        bar    = "#" * filled + "." * (W - filled)
        print(f"{gen:>4}  {gen_best:>5.1f}%  {gen_avg:>5.1f}%  [{bar}]{improved}")

        if best_rate >= 100.0:
            print("\n  Perfect score reached — stopping early.")
            break

        # Next generation: elites + mutations
        elites = [ind for _, ind in scored[:N_ELITE]]
        pop    = list(elites)
        while len(pop) < N_POP:
            pop.append(mutate(random.choice(elites)))

    print(f"\n  Training complete.  Final return rate: {best_rate:.1f}%")
    return best_ind, best_rate


if __name__ == '__main__':
    play_after = '--notplay' not in sys.argv

    print("=" * 52)
    print("  VirtualPC - Evolutionary AI Training")
    print("  Starting from random weights.")
    print("  Each generation plays simulated games and keeps the best.")
    print("=" * 52 + "\n")

    t0 = time.time()
    best_ind, best_rate = train()
    elapsed = time.time() - t0

    mem = Memory(os.path.join(STATE_DIR, 'memory.bin'))
    save(mem, best_ind)
    print(f"  Weights saved to memory.bin  ({elapsed:.1f}s)")
    print(f"  You can now run:  python run_ai_pong.py\n")

    if play_after:
        print("Launching game with trained AI...\n")
        from cpu       import CPU
        from assembler import assemble
        with open(os.path.join(os.path.dirname(__file__), 'programs', 'ai_pong.asm')) as f:
            src = f.read()
        code, origin, labels, errors = assemble(src)
        if errors:
            print("Assembly errors:", errors); sys.exit(1)
        cpu = CPU(mem, turbo=True)
        cpu.load_code(code, origin)
        cpu.PC = origin
        print("Starting (W/S = you  |  right paddle = trained AI  |  Q = quit)\n")
        try:
            cpu.run(max_cycles=999_999_999)
        finally:
            cpu.teardown_display()
        print()
        cpu.show_regs()
        print(f"\nFinal score  You: {mem.read(0xF6)}   AI: {mem.read(0xF7)}")
