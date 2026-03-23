"""
BitWhisker R1 1.x Tkinter – Offline Distilled Engine + Code Sandbox
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pure Python • No API • No external packages
Buttons: black labels on light surfaces (ChatGPT style)
Generation runs in background thread — UI never freezes

Features:
- Distilled offline text engine (fast, direct chat).
- Integrated Python Code Sandbox (Editor & Console).
- Auto-extracts generated code into the sandbox for execution!
"""

import random
import threading
import time
import re
import io
import contextlib
import traceback
import tkinter as tk
from tkinter import scrolledtext, ttk

# ═══════════════════════════════════════════════════════════════════════════════
# ── BitWhisker R1 1.x Distilled Engine (No APIs) ───────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def offline_distilled_stream(prompt, stop_evt):
    """
    A generator that mimics a fast, direct offline model response.
    Yields chunks of text for smooth streaming.
    """
    p = prompt.lower()
    
    # Tiny simulated processing delay
    time.sleep(random.uniform(0.1, 0.3))
    
    full_response = ""
    
    # ── Intent Detection & Simulated Responses ──
    if any(w in p for w in ["hi", "hello", "hey", "greetings", "sup"]):
        full_response = "Hello! I am BitWhisker R1 1.x, operating completely offline. How can I help you with your coding or chat tasks today?"
        
    elif any(w in p for w in ["code", "python", "program", "script", "function", "write", "sort", "prime"]):
        full_response = (
            "Absolutely! Here is a pure Python script for you to test in the Sandbox. "
            "It calculates the first N prime numbers:\n\n"
            "```python\n"
            "def get_primes(n):\n"
            "    \"\"\"Generates the first n prime numbers.\"\"\"\n"
            "    primes = []\n"
            "    num = 2\n"
            "    while len(primes) < n:\n"
            "        is_prime = True\n"
            "        for p in primes:\n"
            "            if p * p > num: break\n"
            "            if num % p == 0:\n"
            "                is_prime = False\n"
            "                break\n"
            "        if is_prime:\n"
            "            primes.append(num)\n"
            "        num += 1\n"
            "    return primes\n\n"
            "# Example usage:\n"
            "print('Calculating first 15 primes...')\n"
            "result = get_primes(15)\n"
            "print(f'Result: {result}')\n"
            "```\n\n"
            "I've loaded this directly into the Sandbox on the right. Click **▶ Run Code** to see it execute!"
        )
        
    elif "joke" in p or "funny" in p:
        jokes = [
            "Why do Python programmers prefer dark mode? Because light attracts bugs!",
            "I'd tell you a UDP joke, but I can't guarantee you'd get it.",
            "There are 10 types of people in the world: those who understand binary, and those who don't.",
            "A programmer puts two glasses on their bedside table before going to sleep. "
            "A full one, in case they get thirsty, and an empty one, in case they don't."
        ]
        full_response = random.choice(jokes)
        
    elif "math" in p or "calculate" in p or "equation" in p:
        full_response = (
            "I am equipped to write algorithms to solve math problems. Ask me to write a "
            "Python function to solve a specific equation, and you can run it directly "
            "in the built-in Sandbox!"
        )
                          
    elif "who are you" in p or "what are you" in p or "version" in p:
        full_response = (
            "I am BitWhisker R1 1.x, a distilled offline model with an integrated Python Sandbox. "
            "I generate direct, helpful responses and code entirely on your local machine."
        )
                          
    else:
        # Fallback reflection
        words = [w for w in p.replace('?', '').replace('.', '').replace(',', '').split() if len(w) > 3]
        if words:
            topic = random.choice(words)
            full_response = (
                f"That's an interesting point regarding '{topic}'. Because I operate on a purely "
                f"offline, algorithmic basis as BitWhisker R1 1.x, my analysis focuses on the semantic structure "
                f"of your text. Try asking me to write some code about {topic}!"
            )
        else:
            full_response = (
                "I am processing your input through my local BitWhisker R1 1.x pathways. "
                "What specific topic or code would you like to explore together?"
            )

    # ── Text Generation Streaming ──
    # Split by spaces but preserve newlines for code blocks
    chunks = full_response.replace('\n', ' \n ').split(' ')
    
    for word in chunks:
        if stop_evt.is_set():
            break
        
        if word == '\n':
            yield '\n'
        elif word == '':
            continue
        else:
            yield word + " "
            
        # Simulate the fast typing speed of a distilled LLM
        time.sleep(random.uniform(0.005, 0.03))


# ═══════════════════════════════════════════════════════════════════════════════
# ── Tkinter GUI ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Colours (ChatGPT dark palette) ─────────────────────────────────────────────
BG_MAIN     = "#212121"   # main chat window
BG_SIDEBAR  = "#171717"   # title bar / headers
BG_INPUT    = "#2f2f2f"   # input box bg
BG_BTN      = "#ececec"   # button surface
FG_BTN      = "#000000"   # black label on buttons
FG_MAIN     = "#ececec"   # primary text
FG_DIM      = "#8e8ea0"   # muted text
FG_USER     = "#ececec"
FG_MODEL    = "#ececec"
ACCENT      = "#10a37f"   # ChatGPT green (send button)
FG_SEND     = "#000000"   # black label on send button

# Sandbox Colours
BG_EDITOR   = "#1e1e1e"   # VS Code dark
BG_CONSOLE  = "#0d0d0d"   # Pitch black terminal
FG_CODE     = "#d4d4d4"   # Light grey code
FG_CONSOLE  = "#4af626"   # Hacker green output

FONT_MAIN   = ("Segoe UI", 10)
FONT_CODE   = ("Courier New", 10)
FONT_TITLE  = ("Segoe UI", 11, "bold")
FONT_SMALL  = ("Segoe UI", 8)

class BitWhiskerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BitWhisker R1 1.x Sandbox")
        # Wider window to accommodate the split pane
        self.geometry("950x500")
        self.resizable(True, True)
        self.configure(bg=BG_SIDEBAR)

        self._busy     = False
        self._stop_evt = threading.Event()

        self._build_ui()
        self._post_system("BitWhisker R1 1.x ready. • Chat & Code Sandbox Active • No APIs")

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Global Title Bar ──
        title_bar = tk.Frame(self, bg=BG_SIDEBAR, height=36)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text="⬡  BitWhisker R1 1.x Workspace",
                 bg=BG_SIDEBAR, fg=FG_MAIN, font=FONT_TITLE,
                 padx=12).pack(side="left", pady=6)

        # ── Split Pane Window ──
        self.paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG_SIDEBAR, bd=0, sashwidth=4)
        self.paned.pack(fill="both", expand=True)

        # Containers
        chat_container = tk.Frame(self.paned, bg=BG_MAIN)
        sandbox_container = tk.Frame(self.paned, bg=BG_SIDEBAR)

        self.paned.add(chat_container, minsize=400, stretch="always")
        self.paned.add(sandbox_container, minsize=350, stretch="always")

        # ══════════════════════════════════════════════════════════════════════
        # 1. LEFT PANE: Chat Area
        # ══════════════════════════════════════════════════════════════════════
        
        # Toolbar inside chat pane
        chat_tools = tk.Frame(chat_container, bg=BG_MAIN, height=30)
        chat_tools.pack(fill="x", side="top")
        for label, cmd in [("↺ Reset Chat", self._reset), ("✕ Clear", self._clear)]:
            tk.Button(chat_tools, text=label, command=cmd,
                      bg=BG_BTN, fg=FG_BTN, activebackground="#d4d4d4",
                      font=FONT_SMALL, relief="flat", cursor="hand2",
                      padx=8, pady=2, bd=0).pack(side="right", padx=4, pady=4)

        self.chat = scrolledtext.ScrolledText(
            chat_container, bg=BG_MAIN, fg=FG_MAIN,
            font=FONT_MAIN, relief="flat", bd=0,
            state="disabled", wrap="word",
            selectbackground="#444", padx=14, pady=10,
            insertbackground=FG_MAIN,
        )
        self.chat.pack(fill="both", expand=True)

        self.chat.tag_config("user_label", foreground=FG_DIM,  font=FONT_SMALL)
        self.chat.tag_config("user_text",  foreground=FG_USER, font=FONT_MAIN)
        self.chat.tag_config("bot_label",  foreground=ACCENT,  font=FONT_SMALL)
        self.chat.tag_config("bot_text",   foreground=FG_MODEL, font=FONT_MAIN)
        self.chat.tag_config("system",     foreground=FG_DIM,  font=FONT_SMALL, justify="center")

        # Input bar
        bar = tk.Frame(chat_container, bg=BG_INPUT, pady=6)
        bar.pack(fill="x", side="bottom")

        self.entry = tk.Text(bar, bg=BG_INPUT, fg=FG_MAIN,
                             insertbackground=FG_MAIN,
                             font=FONT_MAIN, relief="flat", bd=0,
                             height=2, wrap="word",
                             selectbackground="#555")
        self.entry.pack(side="left", fill="x", expand=True, padx=(12, 6), pady=2)
        self.entry.bind("<Return>",       self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)
        self._set_placeholder()

        self.send_btn = tk.Button(
            bar, text="▶", command=self._send, bg=ACCENT, fg=FG_SEND,
            font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2", width=3, bd=0
        )
        self.send_btn.pack(side="right", padx=(0, 10), pady=2)

        self.stop_btn = tk.Button(
            bar, text="■", command=self._stop, bg=BG_BTN, fg=FG_BTN,
            font=("Segoe UI", 12), relief="flat", cursor="hand2", width=3, bd=0, state="disabled"
        )
        self.stop_btn.pack(side="right", padx=(0, 4), pady=2)

        # ══════════════════════════════════════════════════════════════════════
        # 2. RIGHT PANE: Code Sandbox
        # ══════════════════════════════════════════════════════════════════════
        
        sandbox_header = tk.Frame(sandbox_container, bg=BG_SIDEBAR, height=30)
        sandbox_header.pack(fill="x", side="top")
        
        tk.Label(sandbox_header, text="</> Python Sandbox", bg=BG_SIDEBAR, fg=FG_DIM, 
                 font=FONT_SMALL).pack(side="left", padx=8, pady=4)
                 
        tk.Button(sandbox_header, text="▶ Run Code", command=self._run_sandbox,
                  bg=ACCENT, fg=FG_SEND, activebackground="#0d8c6d",
                  font=FONT_SMALL, relief="flat", cursor="hand2",
                  padx=12, pady=2, bd=0).pack(side="right", padx=8, pady=4)

        # Editor
        self.editor = scrolledtext.ScrolledText(
            sandbox_container, bg=BG_EDITOR, fg=FG_CODE,
            font=FONT_CODE, relief="flat", bd=0,
            insertbackground=FG_CODE, selectbackground="#555",
            padx=10, pady=10, height=15
        )
        self.editor.pack(fill="both", expand=True, padx=(0,0), pady=(0,2))
        self.editor.insert("end", "# AI generated code will appear here...\n# You can also write your own code and run it!")

        # Console Output
        console_header = tk.Frame(sandbox_container, bg=BG_SIDEBAR)
        console_header.pack(fill="x")
        tk.Label(console_header, text="Terminal Output", bg=BG_SIDEBAR, fg=FG_DIM, font=FONT_SMALL).pack(side="left", padx=8)

        self.console = scrolledtext.ScrolledText(
            sandbox_container, bg=BG_CONSOLE, fg=FG_CONSOLE,
            font=FONT_CODE, relief="flat", bd=0,
            state="disabled", height=8, padx=10, pady=10
        )
        self.console.pack(fill="x", side="bottom")

    # ── Sandbox Execution ─────────────────────────────────────────────────────

    def _run_sandbox(self):
        """Executes the code in the editor and catches stdout/exceptions."""
        code = self.editor.get("1.0", "end-1c")
        
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.insert("end", "Executing...\n")
        self.update() # Force UI refresh
        
        # Create a string buffer to catch prints
        output_buffer = io.StringIO()
        
        try:
            # Safely redirect stdout to our buffer
            with contextlib.redirect_stdout(output_buffer):
                # Execute the user's code in an isolated dictionary
                exec(code, {})
            result = output_buffer.getvalue()
            if not result.strip():
                result = "[Execution completed with no printed output]"
        except Exception as e:
            # Catch errors and format them nicely
            result = traceback.format_exc()
            
        self.console.delete("1.0", "end")
        self.console.insert("end", result)
        self.console.config(state="disabled")

    # ── Input Handling & Placeholders ─────────────────────────────────────────

    def _set_placeholder(self):
        self.entry.insert("1.0", "Ask for code to run in the sandbox...")
        self.entry.config(fg=FG_DIM)
        self._placeholder_active = True
        self.entry.bind("<FocusIn>",  self._clear_placeholder)
        self.entry.bind("<FocusOut>", self._restore_placeholder)

    def _clear_placeholder(self, _=None):
        if self._placeholder_active:
            self.entry.delete("1.0", "end")
            self.entry.config(fg=FG_MAIN)
            self._placeholder_active = False

    def _restore_placeholder(self, _=None):
        if not self.entry.get("1.0", "end").strip():
            self.entry.insert("1.0", "Ask for code to run in the sandbox...")
            self.entry.config(fg=FG_DIM)
            self._placeholder_active = True

    # ── Chat display helpers ──────────────────────────────────────────────────

    def _append(self, text, tag):
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _post_system(self, msg):
        self._append(f"\n  {msg}\n", "system")

    def _post_user(self, text):
        self._append("\nYou\n", "user_label")
        self._append(f"{text}\n", "user_text")

    def _post_bot_label(self):
        self._append("\nBitWhisker R1 1.x\n", "bot_label")

    def _post_token(self, chunk):
        self._append(chunk, "bot_text")

    def _post_newline(self):
        self._append("\n", "bot_text")

    # ── Send / Stop / Reset / Clear ───────────────────────────────────────────

    def _on_return(self, event):
        if event.state & 0x1: return # Shift+Enter allows newline
        self._send()
        return "break"

    def _send(self):
        if self._busy or self._placeholder_active: return
        text = self.entry.get("1.0", "end").strip()
        if not text: return

        self.entry.delete("1.0", "end")
        if self.focus_get() != self.entry:
            self._restore_placeholder()
            
        self._post_user(text)
        self._start_generation(text)

    def _start_generation(self, prompt):
        self._busy = True
        self._stop_evt.clear()
        self.send_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        t = threading.Thread(target=self._generate_thread, args=(prompt,), daemon=True)
        t.start()

    def _generate_thread(self, prompt):
        self.after(0, self._post_bot_label)
        
        current_response = ""
        
        # Stream the fast, direct response from the offline engine
        for chunk in offline_distilled_stream(prompt, self._stop_evt):
            if self._stop_evt.is_set():
                break
            current_response += chunk
            self.after(0, self._post_token, chunk)

        self.after(0, self._post_newline)
        self.after(0, self._generation_done, current_response)

    def _generation_done(self, final_text=""):
        # Auto-extract Python code from the response and push to Sandbox
        blocks = re.findall(r'```python\n(.*?)\n```', final_text, re.DOTALL)
        if blocks:
            code_content = blocks[-1].strip()
            self.editor.delete("1.0", "end")
            self.editor.insert("end", code_content)
            
            # Clear previous console output to indicate new code is ready
            self.console.config(state="normal")
            self.console.delete("1.0", "end")
            self.console.insert("end", "[New code loaded. Click 'Run Code' to execute]")
            self.console.config(state="disabled")

        self._busy = False
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _stop(self):
        self._stop_evt.set()

    def _reset(self):
        self._stop_evt.set()
        self._busy = False
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._post_system("Engine reset. Awaiting new input.")

    def _clear(self):
        self._stop_evt.set()
        self._busy = False
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._post_system("BitWhisker R1 1.x ready. • Chat & Code Sandbox Active • No APIs")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = BitWhiskerApp()
    app.mainloop()
