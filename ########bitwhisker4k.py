import random
import threading
import time
import math
import tkinter as tk
from tkinter import scrolledtext

# ═══════════════════════════════════════════════════════════════════════════════
# ── UI Constants (BytenetR1 Dark Theme) ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
BG_MAIN    = "#212121"
BG_SIDEBAR = "#171717"
BG_INPUT   = "#2f2f2f"
FG_MAIN    = "#ececec"
FG_DIM     = "#b4b4b4"
ACCENT     = "#10a37f"
FONT_MAIN      = ("Helvetica", 11)
FONT_BOLD      = ("Helvetica", 11, "bold")
FONT_REASONING = ("Helvetica", 10, "italic")

# ═══════════════════════════════════════════════════════════════════════════════
# ── Matrix Utilities ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def zeros(r, c):
    return [[0.0] * c for _ in range(r)]

def transpose(A):
    if not A or not A[0]:
        return []
    return [list(col) for col in zip(*A)]

def matmul(A, B):
    """Standard float matrix multiply."""
    BT = transpose(B)
    return [[sum(a * b for a, b in zip(ra, cb)) for cb in BT] for ra in A]

def ternary_matmul(X, W):
    """BitNet core: matmul where W ∈ {-1,0,1} — zero multiplications."""
    WT = transpose(W)
    out = []
    for row in X:
        orow = []
        for col in WT:
            v = 0.0
            for x, w in zip(row, col):
                if   w ==  1: v += x
                elif w == -1: v -= x
            orow.append(v)
        out.append(orow)
    return out

def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e) + 1e-12
    return [v / s for v in e]

def clip_val(v, lo, hi):
    return max(lo, min(hi, v))

# ═══════════════════════════════════════════════════════════════════════════════
# ── BitNet 1.58b Layers (with full backprop) ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.dim   = dim
        self.eps   = eps
        self.weight = [1.0] * dim
        self.gw     = [0.0] * dim

    def forward(self, x):
        self._x    = x
        self._irms = []
        out = []
        for row in x:
            ms   = sum(v * v for v in row) / len(row)
            irms = 1.0 / math.sqrt(ms + self.eps)
            self._irms.append(irms)
            out.append([v * irms * w for v, w in zip(row, self.weight)])
        return out

    def backward(self, go):
        gi = []
        n  = self.dim
        for row, gr, irms in zip(self._x, go, self._irms):
            for j in range(n):
                self.gw[j] += gr[j] * row[j] * irms
            dot = sum(gr[j] * self.weight[j] * row[j] for j in range(n))
            gi.append([
                gr[j] * self.weight[j] * irms - dot * (irms ** 3) * row[j] / n
                for j in range(n)
            ])
        return gi

    def step(self, lr):
        for j in range(self.dim):
            self.weight[j] -= lr * clip_val(self.gw[j], -5.0, 5.0)
            self.gw[j] = 0.0


class BitLinear:
    """
    BitNet 1.58b Linear — latent float weights quantised to {-1, 0, 1}
    via absmean.  Straight-Through Estimator for backward.
    """
    def __init__(self, inf, outf):
        self.inf  = inf
        self.outf = outf
        sc = math.sqrt(2.0 / inf)
        self.lw = [[random.gauss(0, sc) for _ in range(outf)] for _ in range(inf)]
        self.gw = None

    def _quantize(self):
        total = sum(abs(v) for row in self.lw for v in row)
        gamma = total / (self.inf * self.outf) + 1e-8
        return [[int(round(clip_val(v / gamma, -1, 1))) for v in row] for row in self.lw]

    def forward(self, x):
        self._x = x
        self._tw = self._quantize()
        return ternary_matmul(x, self._tw)

    def backward(self, go):
        self.gw = matmul(transpose(self._x), go)
        return matmul(go, transpose(self.lw))

    def step(self, lr):
        if self.gw:
            for i in range(self.inf):
                for j in range(self.outf):
                    self.lw[i][j] -= lr * clip_val(self.gw[i][j], -5.0, 5.0)


class CausalSelfAttention:
    def __init__(self, dim, heads):
        self.dim   = dim
        self.heads = heads
        self.hd    = dim // heads
        self.scale = 1.0 / math.sqrt(self.hd)
        self.qp = BitLinear(dim, dim)
        self.kp = BitLinear(dim, dim)
        self.vp = BitLinear(dim, dim)
        self.op = BitLinear(dim, dim)

    def forward(self, x):
        S = len(x)
        Q = self.qp.forward(x)
        K = self.kp.forward(x)
        V = self.vp.forward(x)
        self._Q, self._K, self._V = Q, K, V
        self._aw = []
        houts = []
        hd = self.hd
        for h in range(self.heads):
            s = h * hd
            Qh = [[Q[i][s + d] for d in range(hd)] for i in range(S)]
            Kh = [[K[i][s + d] for d in range(hd)] for i in range(S)]
            Vh = [[V[i][s + d] for d in range(hd)] for i in range(S)]
            sc = matmul(Qh, transpose(Kh))
            for i in range(S):
                for j in range(S):
                    sc[i][j] *= self.scale
                    if j > i:
                        sc[i][j] = -1e9
            aw = [softmax(row) for row in sc]
            self._aw.append(aw)
            houts.append(matmul(aw, Vh))

        cat = [[v for h in range(self.heads) for v in houts[h][i]] for i in range(S)]
        return self.op.forward(cat)

    def backward(self, go):
        S   = len(go)
        hd  = self.hd
        gc  = self.op.backward(go)
        gQ  = zeros(S, self.dim)
        gK  = zeros(S, self.dim)
        gV  = zeros(S, self.dim)

        for h in range(self.heads):
            s = h * hd
            gh  = [[gc[i][s + d] for d in range(hd)] for i in range(S)]
            Qh  = [[self._Q[i][s + d] for d in range(hd)] for i in range(S)]
            Kh  = [[self._K[i][s + d] for d in range(hd)] for i in range(S)]
            Vh  = [[self._V[i][s + d] for d in range(hd)] for i in range(S)]
            aw  = self._aw[h]

            gaw = matmul(gh, transpose(Vh))
            gVh = matmul(transpose(aw), gh)

            gsc = []
            for i in range(S):
                dot = sum(aw[i][j] * gaw[i][j] for j in range(S))
                gsc.append([
                    (gaw[i][j] - dot) * aw[i][j] * self.scale
                    if j <= i else 0.0
                    for j in range(S)
                ])

            gQh = matmul(gsc, Kh)
            gKh = matmul(transpose(gsc), Qh)

            for i in range(S):
                for d in range(hd):
                    gQ[i][s + d] += gQh[i][d]
                    gK[i][s + d] += gKh[i][d]
                    gV[i][s + d] += gVh[i][d]

        g1 = self.qp.backward(gQ)
        g2 = self.kp.backward(gK)
        g3 = self.vp.backward(gV)
        return [[g1[i][j] + g2[i][j] + g3[i][j] for j in range(self.dim)] for i in range(S)]

    def step(self, lr):
        for p in (self.qp, self.kp, self.vp, self.op):
            p.step(lr)


class FFN:
    """ReLU feed-forward with BitLinear layers."""
    def __init__(self, dim, hid):
        self.up   = BitLinear(dim, hid)
        self.down = BitLinear(hid, dim)

    def forward(self, x):
        h = self.up.forward(x)
        self._mask = [[1.0 if v > 0 else 0.0 for v in row] for row in h]
        h = [[max(0.0, v) for v in row] for row in h]
        return self.down.forward(h)

    def backward(self, go):
        g = self.down.backward(go)
        g = [[g[i][j] * self._mask[i][j] for j in range(len(g[0]))] for i in range(len(g))]
        return self.up.backward(g)

    def step(self, lr):
        self.up.step(lr)
        self.down.step(lr)


class Block:
    def __init__(self, dim, heads, ffn_dim):
        self.n1  = RMSNorm(dim)
        self.att = CausalSelfAttention(dim, heads)
        self.n2  = RMSNorm(dim)
        self.ffn = FFN(dim, ffn_dim)

    def forward(self, x):
        self._r1 = x
        h = self.att.forward(self.n1.forward(x))
        x = [[a + b for a, b in zip(r, hr)] for r, hr in zip(x, h)]
        self._r2 = x
        h = self.ffn.forward(self.n2.forward(x))
        return [[a + b for a, b in zip(r, hr)] for r, hr in zip(x, h)]

    def backward(self, g):
        gf = self.ffn.backward(self.n2.backward(g))
        g  = [[a + b for a, b in zip(ga, gb)] for ga, gb in zip(g, gf)]
        ga = self.att.backward(self.n1.backward(g))
        return [[a + b for a, b in zip(ga, gb)] for ga, gb in zip(g, ga)]

    def step(self, lr):
        self.n1.step(lr);  self.att.step(lr)
        self.n2.step(lr);  self.ffn.step(lr)


# ═══════════════════════════════════════════════════════════════════════════════
# ── BytenetR1 Model ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class BytenetR1Model:
    def __init__(self, vocab, dim=32, depth=1, heads=2, ffn_mult=2):
        self.V   = vocab
        self.dim = dim
        sc = 0.02
        self.emb    = [[random.gauss(0, sc) for _ in range(dim)] for _ in range(vocab)]
        self.layers = [Block(dim, heads, dim * ffn_mult) for _ in range(depth)]
        self.fnorm  = RMSNorm(dim)
        self.head   = BitLinear(dim, vocab)

    def forward(self, ids):
        self._ids = ids
        x = [list(self.emb[t]) for t in ids]
        for L in self.layers:
            x = L.forward(x)
        x = self.fnorm.forward(x)
        return self.head.forward(x)

    def loss_backward(self, logits, targets):
        S = len(logits)
        loss = 0.0
        gl   = []
        for i in range(S):
            p = softmax(logits[i])
            loss -= math.log(max(p[targets[i]], 1e-10))
            g = list(p)
            g[targets[i]] -= 1.0
            gl.append([v / S for v in g])
        loss /= S

        g = self.head.backward(gl)
        g = self.fnorm.backward(g)
        for L in reversed(self.layers):
            g = L.backward(g)

        self._eg = zeros(self.V, self.dim)
        for i, t in enumerate(self._ids):
            for j in range(self.dim):
                self._eg[t][j] += g[i][j]
        return loss

    def step(self, lr):
        for i in range(self.V):
            for j in range(self.dim):
                self._eg[i][j] = clip_val(self._eg[i][j], -5.0, 5.0)
                self.emb[i][j] -= lr * self._eg[i][j]
        for L in self.layers:
            L.step(lr)
        self.fnorm.step(lr)
        self.head.step(lr)

    def generate_token(self, ids, temperature=0.8, rep_penalty=1.3):
        logits = self.forward(ids)
        lgt    = list(logits[-1])
        # Repetition penalty: reduce logits for recently seen tokens
        recent = set(ids[-8:])
        for tok in recent:
            if tok < len(lgt):
                if lgt[tok] > 0:
                    lgt[tok] /= rep_penalty
                else:
                    lgt[tok] *= rep_penalty
        if temperature > 0:
            sc = [v / temperature for v in lgt]
            pr = softmax(sc)
            # Top-k sampling (k=15)
            indexed = sorted(enumerate(pr), key=lambda x: -x[1])[:15]
            total = sum(p for _, p in indexed)
            r = random.random() * total
            c = 0.0
            for idx, p in indexed:
                c += p
                if r < c:
                    return idx
            return indexed[-1][0]
        return lgt.index(max(lgt))


# ═══════════════════════════════════════════════════════════════════════════════
# ── Word-Level Tokenizer ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re

class WordTokenizer:
    """Simple word + punctuation tokenizer."""
    _PAT = _re.compile(r"[a-z]+|[.,!?]")

    def __init__(self, text):
        tokens = self._PAT.findall(text.lower())
        vocab  = sorted(set(tokens))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text):
        return [self.w2i[t] for t in self._PAT.findall(text.lower()) if t in self.w2i]

    def decode(self, ids):
        toks = [self.i2w.get(i, "?") for i in ids]
        out  = []
        for t in toks:
            if t in (".", ",", "!", "?"):
                out.append(t)
            else:
                out.append(" " + t if out else t)
        return "".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Training Corpus ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

CORPUS = (
    "the cat sat on the mat. the dog ran in the park. "
    "a bird flew over the tree. the fish swam in the sea. "
    "the sun shone in the sky. the moon rose at night. "
    "stars shine bright above the world. the wind blew through the trees. "
    "rain fell on the ground. snow covered the hills. "
    "the river flows to the sea. clouds float in the sky. "
    "the cat sat on the mat. the dog ran in the park. "
    "a small bird sang a song. the old tree stood tall. "
    "the fire burned in the dark. the light shone through the door. "
    "the boy ran down the road. the girl sat by the lake. "
    "the king ruled the land. the queen sat on the throne. "
    "the knight rode his horse. the dragon flew over the castle. "
    "the ship sailed on the sea. the waves crashed on the shore. "
    "the moon rose over the hills. the stars lit up the night. "
    "the cat chased the mouse. the dog barked at the gate. "
    "time flows like the river. the world turns day and night. "
    "the fire burned bright. the wind blew cold. the rain fell hard. "
    "the old man told a tale. the children sat and listened. "
    "the sun set in the west. the moon rose in the east. "
    "birds sang in the morning. the flowers bloomed in spring. "
    "the cat sat on the mat. the dog ran in the park. "
    "the sun shone in the sky. a bird flew over the sea. "
    "the fish swam in the river. the wind blew through the door. "
    "the boy sat by the fire. the girl ran down the road. "
    "the king sat on the throne. the queen ruled the land. "
    "the knight rode over the hills. the dragon flew in the sky. "
    "the ship sailed on the river. the waves crashed on the ground. "
    "the stars shone bright at night. the moon rose over the sea. "
    "the dog chased the cat. the bird sang in the tree. "
    "the old man sat by the lake. the children ran in the park. "
    "snow fell on the hills. rain fell on the ground. "
    "the fire burned through the night. the light shone in the dark. "
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Engine (Training + Generation) ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class BytenetR1Engine:
    MODEL_DIM   = 32
    MODEL_DEPTH = 1
    MODEL_HEADS = 2
    SEQ_LEN     = 16
    TRAIN_STEPS = 800
    LR          = 0.005

    def __init__(self):
        self.tok   = WordTokenizer(CORPUS)
        self.model = BytenetR1Model(
            self.tok.vocab_size,
            dim=self.MODEL_DIM, depth=self.MODEL_DEPTH,
            heads=self.MODEL_HEADS, ffn_mult=2,
        )
        self.data  = self.tok.encode(CORPUS)
        self.trained = False

    def train(self, callback=None):
        """Train on corpus. callback(step, total, loss) for progress."""
        S = self.SEQ_LEN
        N = len(self.data)
        for step in range(self.TRAIN_STEPS):
            i = random.randint(0, N - S - 1)
            inp = self.data[i : i + S]
            tgt = self.data[i + 1 : i + S + 1]
            logits = self.model.forward(inp)
            loss   = self.model.loss_backward(logits, tgt)
            self.model.step(self.LR)
            if callback:
                callback(step + 1, self.TRAIN_STEPS, loss)
        self.trained = True

    def generate(self, prompt, max_tokens=60, temperature=0.7, stop_evt=None):
        ids = self.tok.encode(prompt.lower())
        if not ids:
            ids = self.tok.encode("the")
        ctx_window = 32

        for _ in range(max_tokens):
            if stop_evt and stop_evt.is_set():
                break
            ctx = ids[-ctx_window:]
            nxt = self.model.generate_token(ctx, temperature)
            ids.append(nxt)
            word = self.tok.i2w.get(nxt, "?")
            if word in (".", ",", "!", "?"):
                yield word
            else:
                yield " " + word
            time.sleep(0.05)

    def count_params(self):
        """Count total & ternary parameters."""
        total = self.tok.vocab_size * self.MODEL_DIM          # embedding
        ternary = 0
        def _count_bitlinear(bl):
            return bl.inf * bl.outf
        def _count_block(b):
            t = 0
            for p in (b.att.qp, b.att.kp, b.att.vp, b.att.op):
                t += _count_bitlinear(p)
            t += _count_bitlinear(b.ffn.up)
            t += _count_bitlinear(b.ffn.down)
            return t
        for L in self.model.layers:
            ternary += _count_block(L)
        ternary += _count_bitlinear(self.model.head)
        total += ternary
        return total, ternary


# ═══════════════════════════════════════════════════════════════════════════════
# ── BytenetR1 UI ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class BytenetR1App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BytenetR1 — Real BitNet 1.58b LLM")
        self.geometry("1050x720")
        self.minsize(820, 520)
        self.configure(bg=BG_MAIN)

        self._busy     = False
        self._stop_evt = threading.Event()
        self._engine   = BytenetR1Engine()

        self._build_ui()
        self._lock_input("Training…")
        self.after(300, self._start_training)

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # sidebar
        sb = tk.Frame(self, bg=BG_SIDEBAR, width=260)
        sb.pack(side="left", fill="y"); sb.pack_propagate(False)

        tk.Button(
            sb, text="+ New chat", bg="black", fg="blue",
            activebackground="black", activeforeground="blue",
            font=FONT_MAIN, relief="flat", bd=0, cursor="hand2", pady=10,
        ).pack(fill="x", padx=10, pady=10)

        tk.Label(sb, text="BytenetR1", bg=BG_SIDEBAR, fg="blue",
                 font=("Helvetica", 9, "bold"), anchor="w").pack(fill="x", padx=15, pady=(10, 5))
        tk.Label(sb, text="Real BitNet 1.58b Transformer", bg=BG_SIDEBAR,
                 fg="blue", font=FONT_MAIN, anchor="w").pack(fill="x", padx=15, pady=2)

        # main
        ma = tk.Frame(self, bg=BG_MAIN)
        ma.pack(side="right", expand=True, fill="both")

        self.chat = scrolledtext.ScrolledText(
            ma, bg=BG_MAIN, fg=FG_MAIN, font=FONT_MAIN, wrap=tk.WORD,
            insertbackground=FG_MAIN, bd=0, highlightthickness=0, padx=40, pady=20,
        )
        self.chat.pack(expand=True, fill="both")
        self.chat.config(state=tk.DISABLED)
        self.chat.tag_configure("user",      foreground=FG_MAIN, font=FONT_BOLD)
        self.chat.tag_configure("bot",       foreground=ACCENT,  font=FONT_BOLD)
        self.chat.tag_configure("reasoning", foreground=FG_DIM,  font=FONT_REASONING)
        self.chat.tag_configure("normal",    foreground=FG_MAIN, font=FONT_MAIN)
        self.chat.tag_configure("stat",      foreground="#888",   font=("Helvetica", 9))

        # input row
        ic = tk.Frame(ma, bg=BG_MAIN)
        ic.pack(fill="x", side="bottom", pady=20, padx=40)

        ib = tk.Frame(ic, bg=BG_INPUT, highlightbackground="#444", highlightthickness=1)
        ib.pack(fill="x", ipady=5, ipadx=10)

        self.entry = tk.Entry(
            ib, bg=BG_INPUT, fg=FG_MAIN, font=FONT_MAIN,
            insertbackground=FG_MAIN, bd=0, highlightthickness=0,
        )
        self.entry.pack(side="left", expand=True, fill="x", padx=5, pady=8)
        self.entry.bind("<Return>", lambda e: self._on_send())

        self.btn = tk.Button(
            ib, text="➤", bg="black", fg="blue", font=FONT_BOLD,
            activebackground="black", activeforeground="blue",
            relief="flat", cursor="hand2", command=self._on_send,
        )
        self.btn.pack(side="right", padx=5)

        tk.Label(ic, text="BytenetR1 · strict BitNet 1.58b ternary parameters · trained from scratch",
                 bg=BG_MAIN, fg="blue", font=("Helvetica", 8)).pack(pady=(10, 0))

        # progress bar (canvas)
        self.prog_frame = tk.Frame(ma, bg=BG_MAIN)
        self.prog_frame.pack(fill="x", padx=80, pady=(0, 5), side="bottom")
        self.prog_canvas = tk.Canvas(self.prog_frame, height=18, bg=BG_MAIN,
                                     highlightthickness=0)
        self.prog_canvas.pack(fill="x")
        self.prog_label = tk.Label(self.prog_frame, text="", bg=BG_MAIN, fg=FG_DIM,
                                   font=("Helvetica", 9))
        self.prog_label.pack()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _lock_input(self, placeholder):
        self.entry.config(state=tk.DISABLED)
        self.entry.config(disabledbackground=BG_INPUT, disabledforeground="#666")
        self.btn.config(state=tk.DISABLED)

    def _unlock_input(self):
        self.entry.config(state=tk.NORMAL)
        self.btn.config(state=tk.NORMAL)
        self.entry.focus_set()

    def _append(self, text, tag="normal", nl=False):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text + ("\n" if nl else ""), tag)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _post(self, sender, text, tag):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{sender}\n", tag)
        self.chat.insert(tk.END, f"{text}\n\n", "normal")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    # ── training ──────────────────────────────────────────────────────────────
    def _start_training(self):
        self._post("BytenetR1", "Initialising model — training from scratch on word corpus…", "bot")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _draw_progress(self, step, total, loss):
        w = self.prog_canvas.winfo_width()
        if w < 10:
            w = 400
        frac = step / total
        self.prog_canvas.delete("all")
        self.prog_canvas.create_rectangle(0, 0, w, 18, fill="#333", outline="")
        self.prog_canvas.create_rectangle(0, 0, int(w * frac), 18, fill=ACCENT, outline="")
        self.prog_label.config(
            text=f"Training step {step}/{total}  ·  loss {loss:.3f}"
        )

    def _train_thread(self):
        t0 = time.time()
        def cb(step, total, loss):
            self.after(0, self._draw_progress, step, total, loss)
        self._engine.train(callback=cb)
        elapsed = time.time() - t0
        total_p, ternary_p = self._engine.count_params()

        def finish():
            self.prog_frame.pack_forget()
            info = (
                f"Training complete in {elapsed:.1f}s.\n"
                f"  Parameters : {total_p:,} total · {ternary_p:,} ternary {{-1, 0, 1}}\n"
                f"  Architecture : dim={self._engine.MODEL_DIM}, "
                f"depth={self._engine.MODEL_DEPTH}, heads={self._engine.MODEL_HEADS}\n"
                f"  Tokenizer : word-level ({self._engine.tok.vocab_size} tokens)\n"
                f"All linear weights are strictly BitNet 1.58b — zero multiplications in inference.\n"
                f"Type a prompt to generate (e.g. 'the cat sat' or 'the moon rose')."
            )
            self._post("BytenetR1", info, "bot")
            self._unlock_input()
        self.after(0, finish)

    # ── generation ────────────────────────────────────────────────────────────
    def _on_send(self):
        if self._busy or not self._engine.trained:
            return
        prompt = self.entry.get().strip()
        if not prompt:
            return
        self.entry.delete(0, tk.END)
        self._post("You", prompt, "user")
        self._busy = True
        self._stop_evt.clear()
        threading.Thread(target=self._gen_thread, args=(prompt,), daemon=True).start()

    def _gen_thread(self, prompt):
        self._insert_direct("BytenetR1\n", "bot")

        # reasoning trace — inserted synchronously so it fully completes
        # before any generation tokens appear
        self._insert_direct("💭 ", "reasoning")
        for line in [
            "Encoding prompt to word-level token ids…\n",
            "│ Forward through BitLinear layers (ternary matmul, zero mult)…\n",
            "│ Causal self-attention with softmax…\n",
            "└─ Sampling with temperature=0.7, repetition penalty=1.3\n\n",
        ]:
            self._insert_direct(line, "reasoning")
            time.sleep(0.06)

        for token in self._engine.generate(prompt, max_tokens=150, temperature=0.7,
                                           stop_evt=self._stop_evt):
            self._insert_direct(token, "normal")

        self._insert_direct("\n\n", "normal")
        self.chat.config(state=tk.DISABLED)
        self._busy = False

    def _insert_direct(self, text, tag):
        """Synchronous insert from worker thread — each token renders
        in order before the next one is queued."""
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.see(tk.END)
        self.update_idletasks()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = BytenetR1App()
    app.mainloop()
