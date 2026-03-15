<div align="center">

# 🤖 AURA
### Advanced Unified Reasoning Assistant

**A fully local, voice-driven AI desktop assistant with autonomous agents, computer vision, security tools, OSINT, and a built-in coding IDE — all powered by Ollama.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-black?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?style=flat-square&logo=windows)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📖 What is AURA?

AURA is a **fully local** AI desktop assistant built in Python. It runs entirely on your own machine — no cloud APIs, no subscriptions, no data leaving your computer. You talk to it by voice or text, and it responds through a sleek dark-themed GUI.

Under the hood, AURA is a collection of specialized AI agents that collaborate: a planner breaks your request into steps, a router decides which tools to use, and individual agents execute — whether that's searching the web, writing and running code, controlling your computer, scanning a network, or building an entire software project from scratch.

---

## ✨ Features

### 🗣️ Voice & Chat Interface
- Wake-word free push-to-talk voice input using **Vosk** (offline speech recognition)
- Natural text-to-speech output via **Piper TTS** (zero latency, runs locally)
- Smart Voice Activity Detection (VAD) to auto-stop listening when you finish speaking
- Full chat GUI with message bubbles, obsidian dark theme, and real-time streaming

### 🧠 AI Reasoning
- **Deep Thinking Mode** — uses a dedicated reasoning model (DeepSeek-R1) to work through complex problems step by step
- **LRU-cached thoughts** — identical questions don't re-run the reasoning model
- Multiple Ollama model support: separate models for chat, coding, vision, thinking, and planning

### 🌐 Web Search & Research
- **Web Search** — DuckDuckGo search with top 5 results summarized in context
- **Deep Research** — multi-query research that fires 3+ searches and synthesizes findings
- Automatically triggered when the AI detects your question needs current information

### 🤖 Autonomous Agent Mode
- Give AURA a high-level goal and it **plans and executes a multi-step task chain** without hand-holding
- Supports tool chains: web search → analysis → code generation → save report
- Generates `.docx` Word reports or `.txt` files automatically
- Fallback keyword planner if the LLM planner times out

### 💻 Computer Use
- AURA can **control your computer** — move the mouse, click, type, open apps
- Screenshot → Vision AI → Action Plan → Execute loop
- Supports Chrome, Firefox, Edge, Spotify, Discord, VS Code, and more via shortcut table
- Safe: `pyautogui.FAILSAFE = True` always enabled

### 🖼️ Vision
- **Webcam vision** — grab a frame and describe what AURA sees
- **Real-time YOLO object detection** background thread (~20 FPS)
- Vision results injected into conversation context automatically
- Image results cached via LRU to avoid redundant inference

### 🎨 Image Generation
- Local **SDXL-Turbo** image generation (4-step, ~30 seconds on CPU)
- Auto-fallback to `segmind/tiny-sd` (200MB) if SDXL-Turbo fails
- Pipeline cached in memory — loads once, generates forever
- Built-in Tkinter viewer with Save As / Regenerate buttons

### 🎵 Music Recognition
- Listens to ambient audio and identifies the song playing via **ACRCloud**
- Returns title, artist, and Spotify track ID

### 🔐 Security Agent (Hacker Mode)
Say *"pentest"*, *"run nmap"*, *"security scan"*, or *"hacker mode"* to launch:
- Dedicated terminal GUI with AI-driven penetration testing workflow
- Phases: Reconnaissance → Enumeration → Exploitation Analysis
- Auto-detects best shell: **WSL > Git Bash > PowerShell > Python emulator**
- Pure Python shell emulator for when no real shell is available (ping, curl, whois, dig, nslookup all implemented natively)
- Permission gate — asks before installing any missing tool
- Supports: `nmap`, `gobuster`, `nikto`, `sqlmap`, `hydra`, `ffuf`, `subfinder`, `amass`, `theHarvester`, `whatweb`, `wafw00f`, `sslscan`
- Auto-generates `.docx` penetration test report with findings and AI analysis

### 🕵️ OSINT Engine
Say *"OSINT"*, *"find everything about"*, or *"investigate"* to launch:
- Multi-source person/username/email investigation
- Checks **40+ platforms** for username presence (GitHub, Twitter/X, Reddit, LinkedIn, Instagram, TikTok, and more)
- Generates username permutations from real names
- Email breach checking, domain WHOIS, DNS lookup
- Concurrent execution via `ThreadPoolExecutor`
- Saves full investigation report as `.docx`

### 🏗️ VM Coding Agent (IDE Mode)
Say *"VM mode"*, *"build mode"*, or *"IDE mode"* to launch:
- Describe any software project in plain English
- AI **architect** plans the full file structure, tech stack, and dependencies
- AI **coder** writes every file with full context of already-written files
- Multi-round **auto-fix loop** (up to 8 rounds) — runs the code, reads errors, patches them
- Streaming token output so you watch code being written live
- Linting pass after each file
- Opens a dedicated GUI window with live progress tracking

### 🎼 Spotify Integration
- Browse and control Spotify from a dedicated GUI panel

### 🛠️ Self-Improvement
- AURA can **rewrite its own source files** to improve performance, readability, and error handling
- Syntax validation before overwrite — never saves broken code
- Auto-backup before every change
- Confirmation prompt (can be disabled for batch mode)
- Skips `self_improvement.py` itself (safety guard)

### ⚙️ System Control
- Open and close Windows applications by name
- Click, type, and automate basic UI tasks

---

## 🗂️ Project Structure

```
Aura/
├── config.py
├── aura_config.json
├── aura_memory.json
├── aura.log
├── requirements.txt
├── yolov8n.pt
│
├── ai/
│   ├── agent.py
│   ├── coding.py
│   ├── computer_use.py
│   ├── cursor_overlay.py
│   ├── decision.py
│   ├── hacker_agent.py
│   ├── hacker_runner.py
│   ├── llm.py
│   ├── music_recognition.py
│   ├── planner.py
│   ├── realtime_vision.py
│   ├── thinking.py
│   ├── tool_router.py
│   ├── vision.py
│   ├── vm_agent.py
│   └── vm_runner.py
│
├── core/
│   ├── audio.py
│   ├── memory.py
│   ├── speech.py
│   └── utils.py
│
├── services/
│   ├── performance.py
│   ├── rate_limiter.py
│   └── service_manager.py
│
├── tools/
│   ├── calculator.py
│   ├── executor.py
│   ├── image_gen.py
│   ├── osint.py
│   ├── osint_runner.py
│   ├── self_improvement.py
│   ├── system_control.py
│   └── web_search.py
│
├── ui/
│   ├── display.py
│   ├── hacker_gui.py
│   ├── main_gui.py
│   ├── osint_gui.py
│   ├── spotify_gui.py
│   ├── vm_gui.py
│   └── vm_launch.py
│
└── generated_images/
```

---

## 🔧 Prerequisites

Before installing AURA, you need the following set up:

### 1. Python 3.10 or higher
Download from [python.org](https://www.python.org/downloads/). Make sure to check **"Add Python to PATH"** during install.

### 2. Ollama (Required — runs the AI models)
Download and install from [ollama.com](https://ollama.com). Then pull the models AURA uses:

```bash
# Main conversation model (fast, efficient)
ollama pull gemma3n:e2b

# Deep thinking / reasoning
ollama pull deepseek-r1:8b

# Code generation
ollama pull deepseek-coder-v2:16b

# Vision (webcam / screen reading)
ollama pull qwen3-vl:2b
ollama pull qwen2.5-vl:7b
```

> 💡 You can use different models by editing your `.env` file. Smaller models will be faster but less capable.

### 3. Piper TTS (Required for voice output)
Download from [github.com/rhasspy/piper](https://github.com/rhasspy/piper/releases).

1. Extract it somewhere on your machine (e.g. `C:\piper-tts\piper\`)
2. Download a voice model from the [Piper voices page](https://rhasspy.github.io/piper-samples/)
3. Note the paths — you'll put them in your `.env` file

### 4. Vosk Speech Model (Required for voice input)
Download a model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).

Recommended: `vosk-model-small-en-us-0.15` (~40MB, fast)

Extract it and note the folder path for your `.env` file.

### 5. Optional: WSL (for full Security Agent support on Windows)
```
wsl --install
```
Run in Windows Terminal as Administrator. Restart when prompted. Without WSL, AURA uses a built-in Python shell emulator for security tools.

---

## 🚀 Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/chainspimp/AURA.git
cd AURA
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ `torch` can be large (~2GB). If you only want CPU support and want a faster install:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Step 4 — Create your `.env` file
Create a file named `.env` in the project root:

```env
# ── Paths ────────────────────────────────────────────────────
PIPER_PATH=C:\path\to\piper\piper.exe
PIPER_MODEL=C:\path\to\piper\voices\en_US-hfc_female-medium.onnx
VOSK_MODEL_PATH=C:\path\to\vosk-model-small-en-us-0.15

# ── Ollama Models (change to whatever models you have pulled) ─
OLLAMA_MODEL=gemma3n:e2b
OLLAMA_THINKING_MODEL=deepseek-r1:8b
OLLAMA_CODING_MODEL=deepseek-coder-v2:16b
OLLAMA_VISION_MODEL=qwen3-vl:2b
OLLAMA_COMPUTER_VISION_MODEL=qwen2.5-vl:7b
OLLAMA_COMPUTER_PLAN_MODEL=gemma3n:e2b

# ── Music Recognition (optional) ─────────────────────────────
# Sign up free at acrcloud.com to get these
ACR_HOST=identify-us-west-2.acrcloud.com
ACR_ACCESS_KEY=your_key_here
ACR_ACCESS_SECRET=your_secret_here
```

### Step 5 — Start Ollama
Make sure Ollama is running before launching AURA:
```bash
ollama serve
```
Or just open the Ollama desktop app.

### Step 6 — Launch AURA
```bash
python main_gui.py
```

---

## 🎮 Usage

### Talking to AURA
- **Voice:** Click the microphone button (or press your configured hotkey) and speak. AURA stops listening when you go silent.
- **Text:** Type in the input box and press Enter or the send button.

### Voice Commands & Trigger Phrases

| What you say | What happens |
|---|---|
| *"Search for..."* | Web search via DuckDuckGo |
| *"Research..."* / *"Deep research on..."* | Multi-query deep research |
| *"Write me a Python script that..."* | Code generation, saved to `generated_code/` |
| *"Generate an image of..."* | SDXL-Turbo image generation |
| *"What do you see?"* | Webcam capture + AI description |
| *"Think about..."* / *"Analyse..."* | Deep reasoning mode |
| *"Agent mode: research X and write a report"* | Autonomous multi-step agent |
| *"Pentest 192.168.1.1"* / *"Hacker mode"* | Opens Security Agent terminal |
| *"OSINT on John Smith"* / *"Investigate..."* | Opens OSINT investigation GUI |
| *"VM mode"* / *"Build mode"* / *"IDE mode"* | Opens VM Coding Agent IDE |
| *"What song is this?"* | Music recognition (listens for 8s) |
| *"Open Chrome"* / *"Close Spotify"* | System app control |
| *"Calculate 2^32 / 1024"* | Safe math evaluation |

### Settings
AURA's runtime settings are stored in `aura_config.json` and can be toggled from the GUI:

| Setting | Default | Description |
|---|---|---|
| `voice_enabled` | `true` | Enable/disable TTS voice output |
| `vision_enabled` | `true` | Enable/disable webcam integration |
| `auto_visual_context` | `false` | Auto-capture camera context each turn |
| `enable_thinking` | `false` | Enable deep reasoning by default |
| `enable_web_search` | `true` | Enable automatic web search routing |
| `enable_coding` | `false` | Enable automatic code generation routing |
| `use_vad` | `true` | Use Voice Activity Detection (needs webrtcvad) |
| `response_timeout` | `200` | Ollama response timeout in seconds |
| `max_tts_length` | `500` | Max characters sent to TTS |

---

## 🧩 How the AI Pipeline Works

```
User Input (voice or text)
        │
        ▼
  Decision System ──► Checks: computer use? code? web search? thinking? vision? music?
        │
        ▼
  Tool Execution ──► Runs selected tools in parallel where possible
        │
        ▼
  LLM Response ──► Builds context from memory + tool results → Ollama → response
        │
        ▼
  Memory System ──► Saves interaction with embedding for future semantic recall
        │
        ▼
  TTS Output ──► Piper speaks the response
```

**Memory** uses sentence embeddings (`all-MiniLM-L6-v2`) for semantic search — AURA doesn't just remember the last few messages, it retrieves the most *relevant* past context even from 100 conversations ago.

---

## 🛠️ Troubleshooting

**Ollama connection error**
Make sure `ollama serve` is running. Check it's accessible at `http://localhost:11434`.

**"Vosk model not found"**
Update `VOSK_MODEL_PATH` in your `.env` to point to the extracted model folder (the folder that contains `conf/`, `am/`, etc.).

**"Piper not found" / No voice output**
Update `PIPER_PATH` and `PIPER_MODEL` in your `.env`. Make sure `piper.exe` exists at that path.

**Image generation is very slow**
SDXL-Turbo on CPU takes 20-60 seconds on first run. Subsequent generations are faster because the model stays cached in memory. A GPU (CUDA) will make this 10x faster — set `torch_dtype=torch.float16` and `pipe.to("cuda")` in `image_gen.py`.

**Security Agent shows "virtual shell"**
WSL is not installed or not responding. Install it with `wsl --install` (Admin terminal, requires restart). Without WSL, basic network commands (ping, curl, whois, dig) still work via AURA's built-in Python emulator.

**`webrtcvad` install fails on Windows**
```bash
pip install webrtcvad-wheels
```

**`pyaudio` install fails**
```bash
pip install pipwin
pipwin install pyaudio
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `vosk` | Offline speech recognition |
| `pyaudio` | Microphone audio capture |
| `webrtcvad` | Voice Activity Detection |
| `pygame` | Audio playback |
| `opencv-python` | Webcam frame capture |
| `pillow` | Image processing |
| `sentence-transformers` | Memory embeddings for semantic search |
| `diffusers` + `torch` | SDXL-Turbo image generation |
| `duckduckgo-search` | Web search |
| `requests` | Ollama API calls |
| `numpy` | Embedding math |
| `psutil` | Process management (app control) |
| `pyautogui` | Mouse/keyboard automation |
| `python-dotenv` | `.env` file loading |
| `keyboard` | Hotkey detection |

Optional (installed separately):
| Package | Purpose |
|---|---|
| `python-docx` | Word document generation for reports |
| `ultralytics` | YOLO real-time object detection |
| `sounddevice` | Music recognition audio capture |

---

## 🔒 Privacy & Security

- **100% local** — AURA never sends your conversations, voice data, or files to any external server
- All AI inference runs through your local Ollama instance
- Web search uses DuckDuckGo (no tracking)
- Music recognition uses ACRCloud (audio fingerprint only, not raw audio recording)
- The Security Agent requires explicit permission before installing any tool
- `self_improvement.py` always backs up files before modifying them and validates Python syntax before overwriting

---

## 🗺️ Roadmap

- [ ] Multi-monitor support for computer use
- [ ] Plugin system for custom tools
- [ ] Wake word detection ("Hey AURA")
- [ ] Mobile companion app
- [ ] Multi-agent collaboration (agents spawning sub-agents)
- [ ] GPU-accelerated image generation config in GUI
- [ ] Persistent agent memory across sessions

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Channing Roe**

---

<div align="center">
  <sub>Built with Python, Ollama, and a lot of late nights.</sub>
</div>
