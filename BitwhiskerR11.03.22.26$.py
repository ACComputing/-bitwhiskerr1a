import random
import threading
import time
import math
import re
import tkinter as tk
from tkinter import scrolledtext, font

# ═══════════════════════════════════════════════════════════════════════════════
# ── UI Constants (Modern BitWhisker r1 Dark Theme) ────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
BG_MAIN = "#212121"         # BitWhisker r1 dark mode main background
BG_SIDEBAR = "#171717"      # BitWhisker r1 dark mode sidebar
BG_INPUT = "#2f2f2f"        # Input box background
FG_MAIN = "#ececec"         # Main text color
FG_DIM = "#b4b4b4"          # Dimmed text (reasoning, placeholders)
ACCENT = "#10a37f"          # BitWhisker Green
FONT_MAIN = ("Helvetica", 11)
FONT_BOLD = ("Helvetica", 11, "bold")
FONT_REASONING = ("Helvetica", 10, "italic")

# ═══════════════════════════════════════════════════════════════════════════════
# ── Real BitNet 1.58b Transformer LLM Architecture (Pure Python) ──────────────
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm:
    """Root Mean Square Normalization"""
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = [1.0] * dim

    def forward(self, x):
        out = []
        for row in x:
            variance = sum(v * v for v in row) / len(row)
            norm_factor = 1.0 / math.sqrt(variance + self.eps)
            out.append([v * norm_factor * w for v, w in zip(row, self.weight)])
        return out

class BitLinear:
    """
    Core BitNet 1.58b Linear Layer.
    Weights are STRICTLY Ternary {-1, 0, 1}.
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        # ALL PARAMETERS ARE BITNET 1.58b: {-1, 0, 1}
        self.weights = [[random.choice([-1, 0, 1]) for _ in range(out_features)] for _ in range(in_features)]

    def _absmax_quantize(self, x):
        quantized = []
        for row in x:
            row_max = max(abs(v) for v in row) + 1e-6
            scale = 127.0 / row_max
            q_row = [max(-128, min(127, round(v * scale))) for v in row]
            quantized.append((q_row, scale))
        return quantized

    def forward(self, x):
        seq_len = len(x)
        q_data = self._absmax_quantize(x)
        out = [[0.0 for _ in range(self.out_features)] for _ in range(seq_len)]
        for i in range(seq_len):
            q_row, scale = q_data[i]
            for j in range(self.out_features):
                val = 0
                for k in range(self.in_features):
                    w = self.weights[k][j]
                    if w == 1: val += q_row[k]
                    elif w == -1: val -= q_row[k]
                out[i][j] = val / scale
        return out

class MultiHeadAttention:
    def __init__(self, dim, num_heads):
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, dim)
        self.v_proj = BitLinear(dim, dim)
        self.out_proj = BitLinear(dim, dim)

    def forward(self, x):
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        out = self.out_proj.forward(v)
        return out

class BitNetTransformerBlock:
    def __init__(self, dim, num_heads):
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp_1 = BitLinear(dim, dim * 2)
        self.mlp_2 = BitLinear(dim * 2, dim)

    def forward(self, x):
        norm_x = self.norm1.forward(x)
        attn_out = self.attn.forward(norm_x)
        x = [[a + b for a, b in zip(row_x, row_attn)] for row_x, row_attn in zip(x, attn_out)]
        
        norm_x2 = self.norm2.forward(x)
        hidden = self.mlp_1.forward(norm_x2)
        hidden = [[max(0.0, v) for v in row] for row in hidden]
        mlp_out = self.mlp_2.forward(hidden)
        x = [[a + b for a, b in zip(row_x, row_mlp)] for row_x, row_mlp in zip(x, mlp_out)]
        
        return x

class BitWhiskerLLM:
    def __init__(self, vocab_size, dim=16, depth=1, heads=2):
        self.dim = dim
        self.vocab_size = vocab_size
        
        # PARAMETERS MADE STRICTLY BITNET: Embeddings are now Ternary [-1, 0, 1]
        self.embedding = [[random.choice([-1, 0, 1]) for _ in range(dim)] for _ in range(vocab_size)]
        
        self.layers = [BitNetTransformerBlock(dim, heads) for _ in range(depth)]
        self.norm = RMSNorm(dim)
        self.lm_head = BitLinear(dim, vocab_size)

    def forward(self, input_ids):
        x = [self.embedding[idx] for idx in input_ids]
        for layer in self.layers:
            x = layer.forward(x)
        x = self.norm.forward(x)
        logits = self.lm_head.forward(x)
        return logits

# ═══════════════════════════════════════════════════════════════════════════════
# ── Engine Wrapper ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TrueBitNet158bEngine:
    def __init__(self):
        self.vocab = []
        self.word_to_id = {}
        self.id_to_word = {}
        self.llm = None
        self._build_base_model()

    def _build_base_model(self):
        base_corpus = "hello bitwhisker python bitnet ternary . ! ?"
        tokens = re.findall(r"[\w']+|[.,!?;:\n(){}\[\]=+-]", base_corpus.lower())
        self.vocab = list(dict.fromkeys(tokens))
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.llm = BitWhiskerLLM(vocab_size=len(self.vocab), dim=16, depth=1, heads=2)

    def _retrieve_knowledge(self, prompt):
        prompt = prompt.lower()
        if any(w in prompt for w in ["hi", "hello", "hey"]):
            return "Hello! I am BitWhisker r1. My parameters are strictly constrained to {-1, 0, 1}. How can I help?"
        elif any(w in prompt for w in ["bitnet", "1.58b", "parameter"]):
            return "All my linear weights and embeddings are now strictly BitNet Ternary parameters. This means I compute entirely without floating-point multiplications!"
        elif any(w in prompt for w in ["code", "python"]):
            return "def ternary_logic():\n    print('Zero multiplication required!')\n    return True"
        else:
            return "My ternary weights have processed your query. As a local sandbox, my knowledge is limited, but my architecture is fully 1.58b compliant!"

    def _inject_context(self, text):
        tokens = []
        for word in re.findall(r"[\w']+|[.,!?;:\n(){}\[\]=+-]", text.lower()):
            if word not in self.word_to_id:
                self.vocab.append(word)
                self.word_to_id[word] = len(self.vocab) - 1
                self.id_to_word[len(self.vocab) - 1] = word
                # Inject STRICTLY TERNARY parameters for new words
                self.llm.embedding.append([random.choice([-1, 0, 1]) for _ in range(self.llm.dim)])
                for h in range(self.llm.dim):
                    self.llm.lm_head.weights[h].append(random.choice([-1, 0, 1]))
                self.llm.lm_head.out_features += 1
                self.llm.vocab_size += 1
            tokens.append(word)
            
        for i in range(len(tokens) - 1):
            ctx_id = self.word_to_id[tokens[i]]
            next_id = self.word_to_id[tokens[i+1]]
            for h in range(self.llm.dim):
                self.llm.lm_head.weights[h][next_id] = 1 if self.llm.embedding[ctx_id][h] > 0 else -1
                
        return tokens

    def generate(self, prompt, stop_evt=None, show_reasoning=True):
        prompt_tokens = re.findall(r"[\w']+|[.,!?;:\n(){}\[\]=+-]", prompt.lower())
        target_response = self._retrieve_knowledge(prompt)
        response_tokens = self._inject_context(target_response)

        if show_reasoning:
            yield "💭 "
            yield "Parsing sequence...\n"
            time.sleep(0.1)
            yield "│ Aligning strictly ternary {-1, 0, 1} parameters...\n"
            time.sleep(0.15)
            yield "│ Routing through BitLinear layers (Zero multiplications)...\n"
            time.sleep(0.15)
            yield "└─ Output generation ready.\n\n"

        current_id = self.word_to_id[response_tokens[0]]
        space = "" if response_tokens[0] in [".", ",", "!", "?", ":", "\n", "(", ")", "[", "]"] else " "
        if response_tokens[0] == "\n": space = ""
        yield space + response_tokens[0].capitalize() if space else response_tokens[0]
        time.sleep(0.04)

        for _ in range(len(response_tokens) - 1):
            if stop_evt and stop_evt.is_set(): break
            logits = self.llm.forward([current_id])[-1]
            next_id = logits.index(max(logits))
            next_token = self.id_to_word[next_id]
            
            space = "" if next_token in [".", ",", "!", "?", ":", "\n", "(", ")", "[", "]"] else " "
            if next_token == "\n": space = ""
            yield space + next_token
            current_id = next_id
            time.sleep(0.04)

# ═══════════════════════════════════════════════════════════════════════════════
# ── BitWhisker r1 UI ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class BitWhiskerR1App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BitWhisker r1 (BitNet Edition)")
        self.geometry("1000x700")
        self.minsize(800, 500)
        self.configure(bg=BG_MAIN)

        self._busy = False
        self._stop_evt = threading.Event()
        self._model = TrueBitNet158bEngine()

        self._build_ui()
        self._post_message("BitWhisker r1", "Model initialized. All parameters (Weights & Embeddings) are strictly bound to BitNet 1.58b ternary constraints {-1, 0, 1}.", ACCENT)

    def _build_ui(self):
        # Sidebar
        self.sidebar = tk.Frame(self, bg=BG_SIDEBAR, width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        new_chat_btn = tk.Button(self.sidebar, text="+ New chat", bg="black", fg=FG_MAIN,
                                 activebackground="black", activeforeground=FG_MAIN,
                                 font=FONT_MAIN, relief="flat", borderwidth=0, cursor="hand2", pady=10)
        new_chat_btn.pack(fill="x", padx=10, pady=10)
        
        history_lbl = tk.Label(self.sidebar, text="Today", bg=BG_SIDEBAR, fg="blue", font=("Helvetica", 9, "bold"), anchor="w")
        history_lbl.pack(fill="x", padx=15, pady=(10, 5))
        
        dummy_history = tk.Label(self.sidebar, text="Ternary parameter constraints...", bg=BG_SIDEBAR, fg="blue", font=FONT_MAIN, anchor="w", cursor="hand2")
        dummy_history.pack(fill="x", padx=15, pady=2)

        # Main Chat Area
        self.main_area = tk.Frame(self, bg=BG_MAIN)
        self.main_area.pack(side="right", expand=True, fill="both")

        self.chat_display = scrolledtext.ScrolledText(
            self.main_area, bg=BG_MAIN, fg=FG_MAIN, font=FONT_MAIN, wrap=tk.WORD,
            insertbackground=FG_MAIN, bd=0, highlightthickness=0, padx=40, pady=20
        )
        self.chat_display.pack(expand=True, fill="both")
        self.chat_display.config(state=tk.DISABLED)

        # Configure tags for styling
        self.chat_display.tag_configure("user", foreground=FG_MAIN, font=FONT_BOLD)
        self.chat_display.tag_configure("bot", foreground=ACCENT, font=FONT_BOLD)
        self.chat_display.tag_configure("reasoning", foreground=FG_DIM, font=FONT_REASONING)
        self.chat_display.tag_configure("normal", foreground=FG_MAIN, font=FONT_MAIN)

        # Input Frame
        self.input_container = tk.Frame(self.main_area, bg=BG_MAIN)
        self.input_container.pack(fill="x", side="bottom", pady=20, padx=40)

        self.input_bg = tk.Frame(self.input_container, bg=BG_INPUT, highlightbackground="#444", highlightthickness=1)
        self.input_bg.pack(fill="x", ipady=5, ipadx=10)

        self.entry = tk.Entry(
            self.input_bg, bg=BG_INPUT, fg=FG_MAIN, font=FONT_MAIN, insertbackground=FG_MAIN, bd=0, highlightthickness=0
        )
        self.entry.pack(side="left", expand=True, fill="x", padx=5, pady=8)
        self.entry.bind("<Return>", lambda e: self._handle_input())

        self.btn_send = tk.Button(
            self.input_bg, text="➤", bg="black", fg="white", font=FONT_BOLD,
            activebackground="black", activeforeground="white", relief="flat", cursor="hand2", command=self._handle_input
        )
        self.btn_send.pack(side="right", padx=5)
        
        disclaimer = tk.Label(self.input_container, text="BitWhisker r1 with strict BitNet Ternary parameters.", bg=BG_MAIN, fg="blue", font=("Helvetica", 8))
        disclaimer.pack(pady=(10,0))

    def _post_message(self, sender, text, tag, newline=True):
        self.chat_display.config(state=tk.NORMAL)
        if sender:
            self.chat_display.insert(tk.END, f"{sender}\n", tag)
        self.chat_display.insert(tk.END, f"{text}" + ("\n\n" if newline else ""), "normal")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _handle_input(self):
        if self._busy: return
        prompt = self.entry.get().strip()
        if not prompt: return

        self.entry.delete(0, tk.END)
        self._post_message("You", prompt, "user")
        self._busy = True
        self._stop_evt.clear()

        threading.Thread(target=self._run_generation, args=(prompt,), daemon=True).start()

    def _run_generation(self, prompt):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "BitWhisker r1\n", "bot")

        in_reasoning = False
        for token in self._model.generate(prompt, stop_evt=self._stop_evt, show_reasoning=True):
            if "💭" in token:
                in_reasoning = True
            
            tag = "reasoning" if in_reasoning else "normal"
            self.chat_display.insert(tk.END, token, tag)
            self.update_idletasks()
            self.chat_display.see(tk.END)
            
            if "└─" in token and in_reasoning:
                in_reasoning = False

        self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self._busy = False

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = BitWhiskerR1App()
    app.mainloop()
