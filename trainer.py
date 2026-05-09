"""
trainer.py - Train a neural network to play Pong and save weights to disk.

Network: 3 inputs -> 2 hidden (ReLU) -> 1 output
Weights persist in memory.bin at $D0-$DA between runs.

The fix for gradient cancellation: structured initialization so the two
hidden neurons are NOT symmetric from the start. h1 is pre-wired to detect
"ball above paddle" and h2 to detect "ball below paddle". Gradients flow
immediately and training converges in ~100 epochs.
"""

import random

SCALE   = 4      # quantisation: float 1.0 -> int 4
N_IN    = 3
N_HID   = 2
N_TRAIN = 3000
EPOCHS  = 120
LR      = 0.002  # per-sample online SGD

WEIGHT_ADDR = 0xD0   # base address in data memory

random.seed(7)

def relu(x):   return max(0.0, x)
def relu_d(x): return 1.0 if x > 0 else 0.0

def forward(x, W1, b1, W2, b2):
    pre = [sum(W1[j][i]*x[j] for j in range(N_IN)) + b1[i] for i in range(N_HID)]
    h   = [relu(p) for p in pre]
    out = sum(W2[i]*h[i] for i in range(N_HID)) + b2
    return pre, h, out

def make_data(n):
    data = []
    for _ in range(n):
        ball_y  = random.randint(0, 17)
        p2_y    = random.randint(0, 12)
        ball_dy = random.choice([1, -1])
        centre  = p2_y + 2
        diff    = ball_y - centre
        target  = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        data.append(([float(ball_y >> 1), float(centre >> 1), 1.0 if ball_dy > 0 else 0.0], target))
    return data

def train(data):
    # Structured init: h1 pre-wired for ball_above, h2 for ball_below.
    # This breaks symmetry so gradients flow from epoch 0.
    W1 = [
        [ 0.3, -0.3],   # ball_y:    positive->h1, negative->h2
        [-0.3,  0.3],   # paddle:    negative->h1, positive->h2
        [ 0.0,  0.0],   # ball_dy:   neutral start
    ]
    b1 = [ 0.5, -0.5]   # h1 biased to fire, h2 biased against
    W2 = [ 1.0, -1.0]   # h1 pushes output positive, h2 negative
    b2 = 0.0

    for epoch in range(EPOCHS):
        random.shuffle(data)
        total_loss = 0.0

        for x, tgt in data:
            pre, h, out = forward(x, W1, b1, W2, b2)
            err  = out - tgt
            total_loss += err * err

            # per-sample update with gradient clipping
            d_out = max(-1.0, min(1.0, 2.0 * err))

            for i in range(N_HID):
                W2[i] -= LR * d_out * h[i]
            b2 -= LR * d_out

            for i in range(N_HID):
                d_h = max(-1.0, min(1.0, d_out * W2[i] * relu_d(pre[i])))
                for j in range(N_IN):
                    W1[j][i] -= LR * d_h * x[j]
                b1[i] -= LR * d_h

        if epoch % 30 == 0:
            acc = sum(
                1 for x, t in data
                if (1 if forward(x,W1,b1,W2,b2)[2] > 0 else
                   (-1 if forward(x,W1,b1,W2,b2)[2] < 0 else 0)) == int(t)
            ) / len(data)
            print(f"  epoch {epoch:4d}  loss={total_loss/len(data):.4f}  acc={acc:.1%}")

    return W1, b1, W2, b2

def q(v):
    return max(-128, min(127, int(round(v * SCALE)))) & 0xFF

def weights_to_bytes(W1, b1, W2, b2):
    return [
        q(W1[0][0]), q(W1[1][0]), q(W1[2][0]), q(b1[0]),
        q(W1[0][1]), q(W1[1][1]), q(W1[2][1]), q(b1[1]),
        q(W2[0]),    q(W2[1]),    q(b2),
    ]

def save_weights(mem, weights):
    for i, w in enumerate(weights):
        mem.write(WEIGHT_ADDR + i, w)

def load_weights(mem):
    return [mem.read(WEIGHT_ADDR + i) for i in range(11)]

def weights_are_trained(mem):
    """Returns True if non-default weights are already in memory."""
    w = load_weights(mem)
    return any(b != 0 for b in w)

def print_weights(weights):
    names = ['W(by->h1)','W(pc->h1)','W(dy->h1)','b(h1)',
             'W(by->h2)','W(pc->h2)','W(dy->h2)','b(h2)',
             'W(h1->out)','W(h2->out)','b(out)']
    for i, (w, nm) in enumerate(zip(weights, names)):
        s = w if w < 128 else w - 256
        print(f"  $D{i:X}: {s:4d}  {nm}")

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from memory import Memory
    mem = Memory(os.path.join(os.path.dirname(__file__), '.vpc_state', 'memory.bin'))

    print("Training...")
    data = make_data(N_TRAIN)
    W1, b1, W2, b2 = train(data)
    weights = weights_to_bytes(W1, b1, W2, b2)
    save_weights(mem, weights)
    print("\nWeights saved to memory.bin at $D0-$DA")
    print_weights(weights)
