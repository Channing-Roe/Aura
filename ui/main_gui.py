# ============================================
# FILE: ui/main_gui.py
# AURA — Main Interface  (complete rewrite)
#
# Design: obsidian dark + electric blue
# Layout: strict pack() only, zero mixing
# ============================================

import os, sys, time, threading, subprocess, logging
from datetime import datetime
from typing import Optional, List, Dict, Callable
from aura_startup import bootstrap_aura
bootstrap_aura(your_respond_function)

import tkinter as tk
from tkinter import filedialog

logger = logging.getLogger(__name__)

# ── Colour system ─────────────────────────────────────────
C = {
    # Backgrounds — 5 levels of depth
    "base":     "#060810",
    "surface":  "#0b0f1e",
    "raised":   "#111827",
    "overlay":  "#182033",
    "edge":     "#1e293b",

    # Accents
    "blue":     "#3b82f6",
    "blue_dim": "#1e3a5f",
    "blue_glow":"#60a5fa",
    "cyan":     "#06b6d4",
    "cyan_dim": "#083344",
    "violet":   "#818cf8",
    "green":    "#22c55e",
    "green_dim":"#14532d",
    "amber":    "#f59e0b",
    "red":      "#ef4444",

    # Text
    "t1":  "#f1f5f9",   # primary
    "t2":  "#94a3b8",   # secondary
    "t3":  "#475569",   # muted
    "t4":  "#1e293b",   # ghost

    # Borders
    "b1":  "#1e293b",
    "b2":  "#0f172a",
}

# ── Typography ────────────────────────────────────────────
# All sizes in points — consistent scale
F = {
    "heading":  ("Trebuchet MS", 14, "bold"),
    "subhead":  ("Trebuchet MS", 11, "bold"),
    "body":     ("Segoe UI",     10),
    "body_s":   ("Segoe UI",      9),
    "body_b":   ("Segoe UI",     10, "bold"),
    "mono":     ("Consolas",     10),
    "mono_s":   ("Consolas",      9),
    "mono_b":   ("Consolas",     10, "bold"),
    "icon":     ("Segoe UI",     15),
    "icon_l":   ("Segoe UI",     20),
}

# ── Layout constants ──────────────────────────────────────
SIDEBAR_W   = 64
TOPBAR_H    = 48
STATUSBAR_H = 22
MSG_PAD_X   = 32
MSG_PAD_Y   = 6
BUBBLE_PAD  = 14
INPUT_H     = 72


class MainGUI:

    def __init__(self):
        # Backend — injected by launcher
        self.get_response_fn:  Optional[Callable] = None
        self.listen_fn:        Optional[Callable] = None
        self.tts_fn:           Optional[Callable] = None
        self.memory:           List  = []
        self.app_config:       Dict  = {}
        self.decision_system          = None
        self._save_memory_fn:  Optional[Callable] = None

        # State
        self._processing   = False
        self._listening    = False
        self._attached     = None   # (path, b64) or None
        self._msg_count    = 0
        self._start_time   = time.time()
        self._anim_step    = 0
        self._typing_frame: Optional[tk.Frame] = None

        # Build
        self.root = tk.Tk()
        self.root.title("AURA")
        self.root.configure(bg=C["base"])
        self.root.geometry("1300x820")
        self.root.minsize(900, 600)

        self._build()
        self._start_ticks()
        self.root.after(300, self._welcome)

    # ══════════════════════════════════════════════════════
    # BUILD
    # ══════════════════════════════════════════════════════

    def _build(self):
        # Root: sidebar left, content right — use pack, fill both
        self._sidebar = tk.Frame(self.root, bg=C["base"],
                                  width=SIDEBAR_W, bd=0)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        # Thin separator line
        tk.Frame(self.root, bg=C["b1"], width=1).pack(side="left", fill="y")

        self._content = tk.Frame(self.root, bg=C["base"])
        self._content.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_topbar()
        self._build_chat()
        self._build_input()
        self._build_statusbar()

    # ── Sidebar ───────────────────────────────────────────

    def _build_sidebar(self):
        sb = self._sidebar

        # Logo
        logo = tk.Frame(sb, bg=C["base"], height=TOPBAR_H)
        logo.pack(fill="x")
        logo.pack_propagate(False)
        tk.Label(logo, text="⬡", font=("Segoe UI", 20),
                 fg=C["blue"], bg=C["base"]).place(relx=0.5, rely=0.5, anchor="center")

        # Divider
        tk.Frame(sb, bg=C["b1"], height=1).pack(fill="x")

        # Nav items
        self._nav_btns: Dict[str, tk.Label] = {}
        items = [
            ("💬", "chat",    "Chat",        self._noop),
            ("🖥", "vm",      "VM IDE",      self._launch_vm),
            ("🔒", "hacker",  "Hacker",      self._launch_hacker),
            ("🖱", "pc",      "Computer",    self._launch_computer),
            ("🔍", "osint",   "OSINT",       self._launch_osint),
            ("🎵", "music",   "Music ID",    self._launch_music),
        ]

        for icon, key, tip, cmd in items:
            cell = tk.Frame(sb, bg=C["base"], height=56, cursor="hand2")
            cell.pack(fill="x")
            cell.pack_propagate(False)

            lbl = tk.Label(cell, text=icon,
                           font=F["icon"], fg=C["t3"],
                           bg=C["base"], cursor="hand2")
            lbl.place(relx=0.5, rely=0.5, anchor="center")

            # Hover + active indicator bar on left
            bar = tk.Frame(cell, bg=C["base"], width=3)
            bar.place(x=0, y=8, height=40)

            def _enter(e, l=lbl, b=bar):
                l.config(fg=C["blue_glow"])
                b.config(bg=C["blue"])

            def _leave(e, l=lbl, b=bar, k=key):
                if self._nav_btns.get("active") != k:
                    l.config(fg=C["t3"])
                    b.config(bg=C["base"])

            def _click(e, k=key, c=cmd):
                for key2, (ll, bb) in self._nav_state.items():
                    ll.config(fg=C["t3"])
                    bb.config(bg=C["base"])
                self._nav_state[k][0].config(fg=C["blue_glow"])
                self._nav_state[k][1].config(bg=C["blue"])
                c()

            cell.bind("<Enter>", _enter)
            cell.bind("<Leave>", _leave)
            cell.bind("<Button-1>", _click)
            lbl.bind("<Enter>", _enter)
            lbl.bind("<Leave>", _leave)
            lbl.bind("<Button-1>", _click)

            self._nav_btns[key] = lbl
            if not hasattr(self, "_nav_state"):
                self._nav_state = {}
            self._nav_state[key] = (lbl, bar)

        # Spacer
        tk.Frame(sb, bg=C["base"]).pack(fill="both", expand=True)

        # Settings at bottom
        tk.Frame(sb, bg=C["b1"], height=1).pack(fill="x")
        cfg_cell = tk.Frame(sb, bg=C["base"], height=56, cursor="hand2")
        cfg_cell.pack(fill="x")
        cfg_cell.pack_propagate(False)
        cfg_lbl = tk.Label(cfg_cell, text="⚙", font=F["icon"],
                           fg=C["t3"], bg=C["base"], cursor="hand2")
        cfg_lbl.place(relx=0.5, rely=0.5, anchor="center")
        for w in (cfg_cell, cfg_lbl):
            w.bind("<Enter>", lambda e: cfg_lbl.config(fg=C["t1"]))
            w.bind("<Leave>", lambda e: cfg_lbl.config(fg=C["t3"]))
            w.bind("<Button-1>", lambda e: self._open_settings())

    # ── Top bar ───────────────────────────────────────────

    def _build_topbar(self):
        top = tk.Frame(self._content, bg=C["surface"],
                        height=TOPBAR_H)
        top.pack(fill="x")
        top.pack_propagate(False)

        # Left — name + pulse
        left = tk.Frame(top, bg=C["surface"])
        left.pack(side="left", padx=20, fill="y")

        tk.Label(left, text="AURA",
                 font=F["heading"], fg=C["t1"],
                 bg=C["surface"]).pack(side="left", pady=12)

        self._pulse = tk.Label(left, text="●",
                                font=("Segoe UI", 7),
                                fg=C["green"], bg=C["surface"])
        self._pulse.pack(side="left", padx=(8, 4), pady=12)

        self._status_var = tk.StringVar(value="ready")
        tk.Label(left, textvariable=self._status_var,
                 font=F["body_s"], fg=C["t3"],
                 bg=C["surface"]).pack(side="left", pady=12)

        # Right — model tag + clock
        right = tk.Frame(top, bg=C["surface"])
        right.pack(side="right", padx=20, fill="y")

        self._clock_var = tk.StringVar()
        tk.Label(right, textvariable=self._clock_var,
                 font=F["mono_s"], fg=C["t3"],
                 bg=C["surface"]).pack(side="right", pady=12)

        try:
            from config import OLLAMA_MODEL
            tag = OLLAMA_MODEL.split(":")[0]
        except Exception:
            tag = "local"

        tag_frame = tk.Frame(right, bg=C["blue_dim"])
        tag_frame.pack(side="right", padx=(0, 16), pady=14)
        tk.Label(tag_frame, text=f"  {tag}  ",
                 font=F["mono_s"], fg=C["blue_glow"],
                 bg=C["blue_dim"]).pack(ipady=2)

        # Bottom border
        tk.Frame(self._content, bg=C["b1"], height=1).pack(fill="x")

    # ── Chat area ─────────────────────────────────────────

    def _build_chat(self):
        container = tk.Frame(self._content, bg=C["base"])
        container.pack(fill="both", expand=True)

        # Scrollbar
        self._vsb = tk.Scrollbar(container, orient="vertical",
                                  bg=C["base"], troughcolor=C["base"],
                                  activebackground=C["edge"],
                                  relief="flat", bd=0, width=6)
        self._vsb.pack(side="right", fill="y")

        # Canvas
        self._canvas = tk.Canvas(container, bg=C["base"],
                                  highlightthickness=0,
                                  yscrollcommand=self._vsb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._vsb.config(command=self._canvas.yview)

        # Inner frame
        self._messages = tk.Frame(self._canvas, bg=C["base"])
        self._cwin = self._canvas.create_window(
            (0, 0), window=self._messages, anchor="nw"
        )

        self._messages.bind("<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")
            )
        )
        self._canvas.bind("<Configure>",
            lambda e: self._canvas.itemconfig(self._cwin, width=e.width)
        )
        self._canvas.bind("<MouseWheel>",
            lambda e: self._canvas.yview_scroll(
                int(-1 * e.delta / 120), "units"
            )
        )

    # ── Input bar ─────────────────────────────────────────

    def _build_input(self):
        # Outer wrapper
        outer = tk.Frame(self._content, bg=C["base"])
        outer.pack(fill="x", padx=MSG_PAD_X, pady=(8, 12))

        # Attachment strip (hidden until image attached)
        self._attach_strip = tk.Frame(outer, bg=C["base"], height=0)
        self._attach_strip.pack(fill="x")

        # Input card
        card = tk.Frame(outer, bg=C["raised"],
                         highlightthickness=1,
                         highlightbackground=C["b1"],
                         highlightcolor=C["blue"])
        card.pack(fill="x")

        self._input_card = card

        # Top row: textarea
        self._textarea = tk.Text(
            card,
            height=3,
            font=F["body"],
            bg=C["raised"], fg=C["t1"],
            insertbackground=C["blue"],
            relief="flat", bd=0,
            wrap="word",
            padx=BUBBLE_PAD, pady=10,
            highlightthickness=0,
        )
        self._textarea.pack(fill="x")
        self._textarea.bind("<Return>",       self._on_enter)
        self._textarea.bind("<Shift-Return>", lambda e: None)
        self._textarea.bind("<FocusIn>",
            lambda e: (self._clear_ph(),
                       card.config(highlightbackground=C["blue"])))
        self._textarea.bind("<FocusOut>",
            lambda e: (self._restore_ph(),
                       card.config(highlightbackground=C["b1"])))

        self._ph_text = "Message AURA…   (Enter to send · Shift+Enter for newline)"
        self._ph_active = True
        self._textarea.insert("1.0", self._ph_text)
        self._textarea.config(fg=C["t3"])

        # Divider inside card
        tk.Frame(card, bg=C["b1"], height=1).pack(fill="x")

        # Bottom row: tools left, send right
        bottom = tk.Frame(card, bg=C["raised"])
        bottom.pack(fill="x", padx=10, pady=6)

        # Left buttons
        lbf = tk.Frame(bottom, bg=C["raised"])
        lbf.pack(side="left")

        for icon, tip, cmd in [
            ("📎", "Attach image",     self._attach_image),
            ("🎤", "Voice input",      self._toggle_mic),
        ]:
            btn = self._icon_btn(lbf, icon, cmd)
            btn.pack(side="left", padx=2)

        # Right: send button
        send = tk.Frame(bottom, bg=C["blue"],
                         cursor="hand2")
        send.pack(side="right")
        send.bind("<Button-1>", lambda e: self._send())
        send.bind("<Enter>", lambda e: send.config(bg=C["blue_glow"]))
        send.bind("<Leave>", lambda e: send.config(bg=C["blue"]))

        self._send_inner = tk.Label(
            send, text="  ↑  ",
            font=F["body_b"],
            fg=C["t1"], bg=C["blue"],
            cursor="hand2", padx=4, pady=4,
        )
        self._send_inner.pack()
        self._send_inner.bind("<Button-1>", lambda e: self._send())
        self._send_inner.bind("<Enter>",
            lambda e: self._send_inner.config(bg=C["blue_glow"]))
        self._send_inner.bind("<Leave>",
            lambda e: self._send_inner.config(bg=C["blue"]))

    def _icon_btn(self, parent, icon: str, cmd) -> tk.Label:
        lbl = tk.Label(parent, text=icon,
                        font=("Segoe UI", 12),
                        fg=C["t3"], bg=C["raised"],
                        cursor="hand2", padx=6, pady=2)
        lbl.bind("<Button-1>", lambda e: cmd())
        lbl.bind("<Enter>", lambda e: lbl.config(fg=C["blue_glow"]))
        lbl.bind("<Leave>", lambda e: lbl.config(fg=C["t3"]))
        return lbl

    # ── Status bar ────────────────────────────────────────

    def _build_statusbar(self):
        bar = tk.Frame(self._content, bg=C["surface"],
                        height=STATUSBAR_H)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Frame(bar, bg=C["b1"], height=1).pack(fill="x")

        inner = tk.Frame(bar, bg=C["surface"])
        inner.pack(fill="both", expand=True, padx=16)

        self._stats_var = tk.StringVar(
            value="AURA v2.3  ·  0 messages"
        )
        tk.Label(inner, textvariable=self._stats_var,
                 font=F["mono_s"], fg=C["t3"],
                 bg=C["surface"]).pack(side="left", pady=2)

        self._tool_badge_var = tk.StringVar(value="")
        self._tool_badge = tk.Label(
            inner, textvariable=self._tool_badge_var,
            font=F["mono_s"], fg=C["cyan"],
            bg=C["surface"]
        )
        self._tool_badge.pack(side="right", pady=2)

    # ══════════════════════════════════════════════════════
    # MESSAGE RENDERING
    # ══════════════════════════════════════════════════════

    def _add_message(self, role: str, text: str,
                     image_path: Optional[str] = None,
                     tools_used: Optional[List] = None,
                     response_time: float = 0.0):

        self._msg_count += 1
        is_user  = role == "user"
        ts       = datetime.now().strftime("%H:%M")

        # Outer padding row
        row = tk.Frame(self._messages, bg=C["base"])
        row.pack(fill="x", padx=MSG_PAD_X,
                 pady=(MSG_PAD_Y, MSG_PAD_Y))

        # ── Header row: avatar name + timestamp ──
        hdr = tk.Frame(row, bg=C["base"])
        hdr.pack(fill="x", pady=(0, 4))

        if is_user:
            name_fg, name_txt = C["blue_glow"], "You"
        else:
            name_fg, name_txt = C["violet"], "AURA"

        tk.Label(hdr, text=name_txt,
                 font=F["body_b"],
                 fg=name_fg, bg=C["base"]).pack(side="left")

        tk.Label(hdr, text=f"  {ts}",
                 font=F["mono_s"],
                 fg=C["t3"], bg=C["base"]).pack(side="left")

        # ── Image thumbnail ──
        if image_path and os.path.exists(image_path):
            try:
                from PIL import Image, ImageTk
                img = Image.open(image_path)
                img.thumbnail((260, 180), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_lbl = tk.Label(row, image=photo,
                                    bg=C["raised"], padx=6, pady=6)
                img_lbl.image = photo
                img_lbl.pack(anchor="w", pady=(0, 6))
            except Exception:
                pass

        # ── Bubble ──
        bubble_bg     = C["overlay"] if is_user else C["surface"]
        border_colour = C["blue_dim"] if is_user else C["b1"]

        bubble = tk.Frame(row, bg=bubble_bg,
                           highlightthickness=1,
                           highlightbackground=border_colour)
        bubble.pack(fill="x")

        txt_widget = tk.Text(
            bubble,
            font=F["body"],
            bg=bubble_bg,
            fg=C["t1"],
            relief="flat", bd=0,
            wrap="word",
            state="normal",
            cursor="arrow",
            padx=BUBBLE_PAD, pady=BUBBLE_PAD,
            highlightthickness=0,
            spacing1=2, spacing3=2,
        )

        # Code block highlighting
        self._insert_text(txt_widget, text)

        # Compute auto-height
        lines   = text.count('\n') + 1
        wrapped = sum(
            max(1, (len(l) // 95) + 1)
            for l in text.split('\n')
        )
        txt_widget.config(height=min(max(lines, wrapped) + 1, 50))
        txt_widget.config(state="disabled")
        txt_widget.pack(fill="x")

        # ── Tool badges ──
        if tools_used:
            icons_map = {
                "thinking_used":   ("🧠", "Thinking"),
                "web_used":        ("🌐", "Web"),
                "research_used":   ("📚", "Research"),
                "vision_used":     ("👁",  "Vision"),
                "image_generated": ("🎨", "Image"),
                "code_generated":  ("💻", "Code"),
                "computer_used":   ("🖥",  "Computer"),
            }
            tag_row = tk.Frame(bubble, bg=bubble_bg)
            tag_row.pack(fill="x", padx=BUBBLE_PAD, pady=(0, 8))
            for t in tools_used:
                if t in icons_map:
                    ico, lbl = icons_map[t]
                    pill = tk.Frame(tag_row, bg=C["cyan_dim"])
                    pill.pack(side="left", padx=(0, 4))
                    tk.Label(pill,
                             text=f" {ico} {lbl} ",
                             font=F["mono_s"],
                             fg=C["cyan"],
                             bg=C["cyan_dim"]).pack(ipady=1)

        # ── Response time ──
        if response_time > 0 and not is_user:
            tk.Label(row,
                     text=f"{response_time:.1f}s",
                     font=F["mono_s"],
                     fg=C["t3"], bg=C["base"]).pack(anchor="e", pady=(2, 0))

        self.root.after(60, self._scroll_bottom)
        self._update_stats()

    def _insert_text(self, widget: tk.Text, text: str):
        """Insert text with basic code block detection."""
        import re

        widget.tag_configure("code_bg",
                              background=C["base"],
                              foreground=C["cyan"],
                              font=F["mono"],
                              lmargin1=BUBBLE_PAD,
                              lmargin2=BUBBLE_PAD)
        widget.tag_configure("normal",
                              foreground=C["t1"],
                              font=F["body"])

        parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                inner = re.sub(r'^```\w*\n?', '', part)
                inner = re.sub(r'\n?```$', '', inner)
                widget.insert("end", inner, "code_bg")
            else:
                widget.insert("end", part, "normal")

    def _add_typing(self) -> tk.Frame:
        frame = tk.Frame(self._messages, bg=C["base"])
        frame.pack(fill="x", padx=MSG_PAD_X, pady=MSG_PAD_Y)

        tk.Label(frame, text="AURA",
                 font=F["body_b"],
                 fg=C["violet"], bg=C["base"]).pack(anchor="w", pady=(0, 4))

        bubble = tk.Frame(frame, bg=C["surface"],
                           highlightthickness=1,
                           highlightbackground=C["b1"])
        bubble.pack(anchor="w")

        self._dot_lbl = tk.Label(bubble,
                                  text="   ●   ○   ○   ",
                                  font=F["body"],
                                  fg=C["blue"],
                                  bg=C["surface"],
                                  padx=BUBBLE_PAD, pady=BUBBLE_PAD)
        self._dot_lbl.pack()

        self._dot_step = 0
        self._animate_dots()
        self.root.after(60, self._scroll_bottom)
        return frame

    def _animate_dots(self):
        if not hasattr(self, '_dot_lbl') or \
                not self._dot_lbl.winfo_exists():
            return
        pats = [
            ("   ●   ○   ○   ", C["blue"]),
            ("   ○   ●   ○   ", C["violet"]),
            ("   ○   ○   ●   ", C["cyan"]),
        ]
        try:
            txt, col = pats[self._dot_step % 3]
            self._dot_lbl.config(text=txt, fg=col)
            self._dot_step += 1
        except Exception:
            return
        self.root.after(380, self._animate_dots)

    def _scroll_bottom(self):
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    # ══════════════════════════════════════════════════════
    # INPUT HANDLING
    # ══════════════════════════════════════════════════════

    # Placeholder
    def _clear_ph(self):
        if self._ph_active:
            self._textarea.delete("1.0", "end")
            self._textarea.config(fg=C["t1"])
            self._ph_active = False

    def _restore_ph(self):
        if not self._textarea.get("1.0", "end").strip():
            self._textarea.insert("1.0", self._ph_text)
            self._textarea.config(fg=C["t3"])
            self._ph_active = True

    def _on_enter(self, event):
        if event.state & 0x1:   # Shift — allow newline
            return
        self._send()
        return "break"

    def _send(self):
        if self._processing:
            return
        raw = self._textarea.get("1.0", "end").strip()
        if not raw or raw == self._ph_text:
            return

        # Clear input
        self._textarea.delete("1.0", "end")
        self._restore_ph()

        image_path = self._attached[0] if self._attached else None
        self._clear_attachment()

        self._add_message("user", raw, image_path=image_path)
        self._processing = True
        self._set_status("thinking…", C["amber"])

        typing = self._add_typing()
        threading.Thread(
            target=self._process,
            args=(raw, image_path, typing),
            daemon=True
        ).start()

    def _process(self, text: str, image_path: Optional[str],
                  typing: tk.Frame):
        t0  = time.time()
        resp = ""
        meta: Dict = {}

        try:
            from ai.hacker_runner import should_launch_hacker
            from ai.vm_runner import should_launch_vm

            if should_launch_vm(text):
                self.root.after(0, typing.destroy)
                self.root.after(0, self._launch_vm)
                resp = "🖥  Opening VM IDE…"

            elif should_launch_hacker(text):
                self.root.after(0, typing.destroy)
                self.root.after(0, self._launch_hacker)
                resp = "🔒 Opening Security Terminal…"

            elif self.get_response_fn:
                if image_path:
                    text = f"[Image: {os.path.basename(image_path)}]\n{text}"
                resp, meta = self.get_response_fn(
                    text, self.memory,
                    decision_system=self.decision_system,
                    app_config=self.app_config,
                )
            else:
                resp = "AURA backend not connected. Run via main_gui_launch.py"

        except Exception as e:
            logger.error(f"Process error: {e}", exc_info=True)
            resp = f"Something went wrong: {e}"

        elapsed = time.time() - t0

        def _show():
            try:
                typing.destroy()
            except Exception:
                pass
            tools = meta.get("tools_used", []) if meta else []
            self._add_message("aura", resp,
                               tools_used=tools,
                               response_time=elapsed)
            if self.tts_fn and self.app_config.get("voice_enabled", True):
                threading.Thread(
                    target=self.tts_fn,
                    args=(resp[:500], self.app_config),
                    daemon=True
                ).start()
            if self._save_memory_fn:
                try:
                    self._save_memory_fn(text, resp, meta)
                except Exception:
                    pass
            self._processing = False
            self._set_status("ready", C["green"])

        self.root.after(0, _show)

    # ══════════════════════════════════════════════════════
    # VOICE
    # ══════════════════════════════════════════════════════

    def _toggle_mic(self):
        if self._listening:
            self._listening = False
            self._set_status("ready", C["green"])
            return
        self._listening = True
        self._set_status("listening…", C["red"])
        threading.Thread(target=self._do_listen, daemon=True).start()

    def _do_listen(self):
        try:
            if self.listen_fn:
                text, ok = self.listen_fn()
                if ok and text:
                    def _fill():
                        self._clear_ph()
                        self._textarea.delete("1.0", "end")
                        self._textarea.config(fg=C["t1"])
                        self._textarea.insert("1.0", text)
                        self._ph_active = False
                    self.root.after(0, _fill)
        except Exception as e:
            logger.error(f"Listen: {e}")
        finally:
            self._listening = False
            self.root.after(0, lambda: self._set_status("ready", C["green"]))

    # ══════════════════════════════════════════════════════
    # ATTACHMENT
    # ══════════════════════════════════════════════════════

    def _attach_image(self):
        path = filedialog.askopenfilename(
            title="Attach Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.webp *.bmp"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self._attached = (path, None)

        # Show strip
        for w in self._attach_strip.winfo_children():
            w.destroy()

        strip = tk.Frame(self._attach_strip, bg=C["raised"])
        strip.pack(fill="x", pady=(0, 6))

        try:
            from PIL import Image, ImageTk
            img = Image.open(path)
            img.thumbnail((36, 36), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            il = tk.Label(strip, image=ph, bg=C["raised"])
            il.image = ph
            il.pack(side="left", padx=(8, 6), pady=6)
        except Exception:
            tk.Label(strip, text="📎", font=F["body"],
                     fg=C["t2"], bg=C["raised"]).pack(
                side="left", padx=8, pady=6)

        tk.Label(strip,
                 text=os.path.basename(path)[:48],
                 font=F["body_s"], fg=C["t2"],
                 bg=C["raised"]).pack(side="left", pady=6)

        tk.Button(strip, text="✕",
                   font=F["body_s"], fg=C["t3"],
                   bg=C["raised"], relief="flat", bd=0,
                   cursor="hand2",
                   command=self._clear_attachment
                   ).pack(side="right", padx=10, pady=6)

    def _clear_attachment(self):
        self._attached = None
        for w in self._attach_strip.winfo_children():
            w.destroy()

    # ══════════════════════════════════════════════════════
    # LAUNCHERS
    # ══════════════════════════════════════════════════════

    def _noop(self): pass

    def _launch_vm(self):
        self._badge("🖥  VM IDE")
        try:
            from ai.vm_runner import launch_vm_mode
            launch_vm_mode()
        except Exception as e:
            self._add_message("aura", f"VM launch failed: {e}")

    def _launch_hacker(self):
        self._badge("🔒  Hacker")
        try:
            launcher = os.path.join(os.path.dirname(__file__), "hacker_launch.py")
            if os.path.exists(launcher):
                subprocess.Popen([sys.executable, launcher])
            else:
                from ai.hacker_runner import launch_hacker_mode
                threading.Thread(
                    target=lambda: launch_hacker_mode(blocking=True),
                    daemon=True
                ).start()
        except Exception as e:
            self._add_message("aura", f"Hacker launch failed: {e}")

    def _launch_computer(self):
        self._badge("🖱  Computer Use")
        self._add_message(
            "aura",
            "Computer use is active. Just tell me what to do — "
            "I'll move the cursor and interact with your screen."
        )

    def _launch_osint(self):
        self._badge("🔍  OSINT")
        try:
            from ui.osint_gui import launch_osint_gui
            threading.Thread(
                target=lambda: launch_osint_gui(blocking=True),
                daemon=True
            ).start()
        except Exception as e:
            self._add_message("aura", f"OSINT launch failed: {e}")

    def _launch_music(self):
        self._badge("🎵  Music ID")
        self._add_message("aura", "Listening for music… 🎵")
        threading.Thread(target=self._do_music, daemon=True).start()

    def _do_music(self):
        try:
            result = {"title": "Unknown", "artist": "Unknown"}
            self.root.after(0, lambda: self._add_message(
                "aura",
                f"🎵 {result['title']} — {result['artist']}"
            ))
        except Exception as e:
            self.root.after(0, lambda: self._add_message(
                "aura", f"Music ID failed: {e}"
            ))

    def _badge(self, text: str):
        self._tool_badge_var.set(text)
        self.root.after(4000, lambda: self._tool_badge_var.set(""))

    # ══════════════════════════════════════════════════════
    # SETTINGS
    # ══════════════════════════════════════════════════════

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("400x380")
        win.resizable(False, False)
        win.configure(bg=C["surface"])
        win.grab_set()

        tk.Label(win, text="Settings",
                 font=F["heading"], fg=C["t1"],
                 bg=C["surface"]).pack(padx=28, pady=(22, 4), anchor="w")

        tk.Frame(win, bg=C["b1"], height=1).pack(fill="x", padx=28, pady=(4, 12))

        toggles = [
            ("Voice responses",     "voice_enabled"),
            ("Auto vision context", "auto_visual_context"),
            ("Web search",          "enable_web_search"),
            ("Deep thinking",       "enable_thinking"),
            ("Code generation",     "enable_coding"),
            ("Debug mode",          "debug_mode"),
        ]

        for label, key in toggles:
            row = tk.Frame(win, bg=C["surface"])
            row.pack(fill="x", padx=28, pady=5)

            tk.Label(row, text=label,
                     font=F["body"], fg=C["t2"],
                     bg=C["surface"]).pack(side="left")

            var = tk.BooleanVar(value=self.app_config.get(key, True))

            def _toggle(k=key, v=var):
                self.app_config[k] = v.get()
                try:
                    from config import save_config
                    save_config(self.app_config)
                except Exception:
                    pass

            tk.Checkbutton(
                row, variable=var,
                bg=C["surface"],
                fg=C["blue"],
                selectcolor=C["edge"],
                activebackground=C["surface"],
                relief="flat", bd=0,
                command=_toggle,
            ).pack(side="right")

        tk.Frame(win, bg=C["b1"], height=1).pack(fill="x", padx=28, pady=12)

        close = tk.Frame(win, bg=C["blue"], cursor="hand2")
        close.pack(ipadx=28, ipady=6)
        close.bind("<Button-1>", lambda e: win.destroy())
        tk.Label(close, text="  Close  ",
                 font=F["body_b"], fg=C["t1"],
                 bg=C["blue"], cursor="hand2",
                 padx=8, pady=4).pack()

    # ══════════════════════════════════════════════════════
    # STATUS / HELPERS
    # ══════════════════════════════════════════════════════

    def _set_status(self, text: str, colour: str = None):
        colour = colour or C["green"]
        self._status_var.set(text)
        self._pulse.config(fg=colour)

    def _update_stats(self):
        uptime = int(time.time() - self._start_time)
        h, rem = divmod(uptime, 3600)
        m, s   = divmod(rem, 60)
        self._stats_var.set(
            f"AURA v2.3  ·  {self._msg_count} messages  "
            f"·  {h:02d}:{m:02d}:{s:02d}"
        )

    def _start_ticks(self):
        def tick():
            self._clock_var.set(datetime.now().strftime("%H:%M:%S"))
            self._update_stats()
            self.root.after(1000, tick)
        tick()

    def _welcome(self):
        self._add_message(
            "aura",
            "Hey — I'm AURA, your local AI assistant.\n\n"
            "Chat with me below, attach images with 📎, or use the "
            "sidebar to open a tool:\n\n"
            "  🖥  VM IDE  —  build full projects with live code streaming\n"
            "  🔒  Hacker  —  security scanning and pentesting\n"
            "  🖱  Computer Use  —  I control your mouse and keyboard\n"
            "  🔍  OSINT  —  open-source intelligence\n"
            "  🎵  Music ID  —  identify any song playing nearby\n\n"
            "What can I help you with?"
        )

    def show(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════
# WIRED LAUNCHER
# ══════════════════════════════════════════════════════════

def create_gui_with_backend(
    get_response_fn, listen_fn, tts_fn,
    memory, app_config, decision_system,
    save_memory_fn=None,
) -> MainGUI:
    gui = MainGUI()
    gui.get_response_fn = get_response_fn
    gui.listen_fn       = listen_fn
    gui.tts_fn          = tts_fn
    gui.memory          = memory
    gui.app_config      = app_config
    gui.decision_system = decision_system
    if save_memory_fn:
        gui._save_memory_fn = save_memory_fn
    return gui
