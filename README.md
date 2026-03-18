<div align="center">

# 🤖 AURA
### Advanced Unified Reasoning Assistant

**A fully local, voice-driven AI desktop assistant with autonomous agents, computer vision, security tools, OSINT, a built-in coding IDE, browser automation, a plugin skill system, and Telegram remote access — all powered by Ollama.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-black?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-Source%20Available-orange?style=flat-square)

</div>

---

## 📖 What is AURA?

AURA is a **fully local** AI desktop assistant built in Python. It runs entirely on your own machine — no cloud APIs, no subscriptions, no data leaving your computer. You talk to it by voice or text, and it responds through a sleek dark-themed GUI.

Under the hood, AURA is a collection of specialised AI agents that collaborate: a planner breaks your request into steps, a router decides which tools to use, and individual agents execute — whether that's searching the web, writing and running code, controlling your computer, scanning a network, building an entire software project from scratch, or running a scheduled background task while you sleep.

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
- **Web Search** — DuckDuckGo search with top 5 results summarised in context
- **Deep Research** — multi-query research that fires 3+ searches and synthesises findings
- **Browser Automation** — Playwright-powered headless browser for real page interaction: fill forms, log in to sites, scrape dynamic content (no more fragile screenshot→click loops)
- Automatically triggered when AURA detects your question needs current information

### 🔧 Skills Plugin System *(NEW)*
- Drop a folder into `skills/` and AURA auto-loads it on next start — no config needed
- Skills run **before** the LLM is called, giving instant results for common tasks
- Ships with **Weather** (wttr.in, no API key) and **Unit Converter** built in
- Full authoring guide in `skills/HOW_TO_WRITE_A_SKILL.md` — open to community contributions
- Enable/disable individual skills at runtime from `aura_config.json`

### 📅 Proactive Scheduler *(NEW)*
- AURA can act **without you prompting it** — check emails, run research, send reminders
- Natural language scheduling: *"remind me every day at 9am"*, *"search for AI news every Monday at 8am"*, *"in 15 minutes..."*
- Jobs persist in `scheduler_jobs.json` and survive restarts
- View scheduled tasks from the GUI or via Telegram `/schedule`

### 📱 Telegram Remote Interface *(NEW)*
- Talk to AURA from your phone anywhere via **Telegram**
- Supports text messages, voice notes (auto-transcribed via Vosk), and photos (vision)
- Commands: `/skills`, `/schedule`, `/memory`, `/status`
- Locked to your Telegram user ID — no one else can access your instance
- Setup: add `TELEGRAM_TOKEN` and `TELEGRAM_ALLOWED_IDS` to `.env`

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
- SDXL-Turbo local image generation — fully offline
- Images saved to `generated_images/`
- GPU acceleration supported (CUDA)

### 🔐 Security Agent
- Full pentest terminal with AI guidance
- Supports nmap, gobuster, hashcat, and more
- WSL integration on Windows; native shell on Linux/macOS

### 🕵️ OSINT Agent
- Investigate usernames, emails, names across dozens of platforms
- Generates full `.docx` intelligence reports
- GitHub, Reddit, HackerNews, and more via free APIs

### 💻 VM Coding Agent (IDE Mode)
- Describe a project in plain English → AURA architects, writes, and runs it
- DeepSeek-Coder-V2 for file generation with multi-round fix loops
- Built-in code editor with syntax highlighting and live terminal

### 🎵 Music Recognition
- ACRCloud-powered music identification (listens for 8 seconds)

### 🧬 Self-Improvement
- AURA can read, critique, and rewrite its own source files
- Always backs up before modifying, validates Python syntax before overwriting

---

## 🖥️ Platform Support

| Platform | Status |
|---|---|
| Windows 10/11 | ✅ Full support |
| Linux (Ubuntu, Arch, Fedora) | ✅ Full support (v2) |
| macOS (Intel + Apple Silicon) | ✅ Full support (v2) |

---

## 🚀 Installation

### Windows

**Step 1 — Install Ollama**
Download from [ollama.com](https://ollama.com) and pull your models:
```bash
ollama pull gemma3n:e2b
ollama pull deepseek-r1:8b
ollama pull deepseek-coder-v2:16b
ollama pull qwen2.5-vl:7b
```

**Step 2 — Clone the repo**
```bash
git clone https://github.com/Channing-Roe/AURA.git
cd AURA
```

**Step 3 — Install Python dependencies**
```bash
pip install -r requirements.txt
playwright install chromium
```

**Step 4 — Set up Vosk speech model**

Download [vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip), extract it, and note the folder path.

**Step 5 — Configure `.env`**
```env
PIPER_PATH=C:\path\to\piper.exe
PIPER_MODEL=C:\path\to\voices\en_US-hfc_female-medium.onnx
VOSK_MODEL_PATH=C:\path\to\vosk-model-small-en-us-0.15
OLLAMA_MODEL=gemma3n:e2b

# Optional — Telegram remote access
TELEGRAM_TOKEN=your_bot_token_from_botfather
TELEGRAM_ALLOWED_IDS=your_telegram_user_id
```

**Step 6 — Launch AURA**
```bash
python main_gui.py
```

---

### Linux / macOS

One-command setup (installs all system deps, Piper TTS, Vosk model, Playwright):
```bash
bash setup_unix.sh
```

Then launch:
```bash
python main_gui.py
```

---

## 🎮 Usage

### Talking to AURA
- **Voice:** Click the microphone button and speak. AURA stops listening when you go silent.
- **Text:** Type in the input box and press Enter or the send button.
- **Telegram:** Message your bot from your phone — same capabilities, remote access.

### Voice Commands & Trigger Phrases

| What you say | What happens |
|---|---|
| *"Search for..."* | Web search via DuckDuckGo |
| *"Research..."* / *"Deep research on..."* | Multi-query deep research |
| *"Go to github.com and..."* | Playwright browser automation |
| *"Fill in the form at..."* | Browser form filling |
| *"Write me a Python script that..."* | Code generation, saved to `generated_code/` |
| *"Generate an image of..."* | SDXL-Turbo image generation |
| *"What do you see?"* | Webcam capture + AI description |
| *"Think about..."* / *"Analyse..."* | Deep reasoning mode |
| *"Agent mode: research X and write a report"* | Autonomous multi-step agent |
| *"Remind me every day at 9am to..."* | Creates a scheduled task |
| *"Every Monday at 8am, search for AI news"* | Recurring scheduled research |
| *"Pentest 192.168.1.1"* / *"Hacker mode"* | Opens Security Agent terminal |
| *"OSINT on John Smith"* / *"Investigate..."* | Opens OSINT investigation GUI |
| *"VM mode"* / *"Build mode"* / *"IDE mode"* | Opens VM Coding Agent IDE |
| *"What song is this?"* | Music recognition (listens for 8s) |
| *"Open Chrome"* / *"Close Spotify"* | System app control |
| *"Calculate 2^32 / 1024"* | Safe math evaluation |
| *"Weather in London"* | Weather skill (instant, no LLM) |
| *"100 km in miles"* | Unit converter skill (instant) |

### Settings
AURA's runtime settings are stored in `aura_config.json`:

| Setting | Default | Description |
|---|---|---|
| `voice_enabled` | `true` | Enable/disable TTS voice output |
| `vision_enabled` | `true` | Enable/disable webcam integration |
| `auto_visual_context` | `false` | Auto-capture camera context each turn |
| `enable_thinking` | `false` | Enable deep reasoning by default |
| `enable_web_search` | `true` | Enable automatic web search routing |
| `enable_coding` | `false` | Enable automatic code generation routing |
| `enable_skills` | `true` | Enable skills plugin system |
| `enable_browser` | `true` | Enable Playwright browser automation |
| `enable_scheduler` | `true` | Enable proactive scheduler daemon |
| `enable_telegram_bot` | `false` | Enable Telegram remote interface |
| `browser_headless` | `true` | Run browser without visible window |
| `use_vad` | `true` | Use Voice Activity Detection |
| `response_timeout` | `200` | Ollama response timeout in seconds |
| `max_tts_length` | `500` | Max characters sent to TTS |

---

## 🧩 How the AI Pipeline Works

```
User Input (voice, text, or Telegram)
        │
        ▼
  Skills Fast-Path ──► Matches keyword? → Run skill instantly (no LLM)
        │ no match
        ▼
  Scheduler Check ──► Schedule phrase? → Create APScheduler job
        │ no match
        ▼
  Decision System ──► computer use? code? browser? web search? thinking? vision?
        │
        ▼
  Tool Execution ──► Runs selected tools (browser, search, vision, etc.)
        │
        ▼
  LLM Response ──► Builds context from memory + tool results → Ollama → response
        │
        ▼
  Memory System ──► Saves interaction with embedding for future semantic recall
        │
        ▼
  TTS + GUI Output ──► Piper speaks the response, GUI displays it
        │
        ▼
  Telegram Mirror ──► Response also sent to Telegram if message came from there
```

**Memory** uses sentence embeddings (`all-MiniLM-L6-v2`) for semantic search — AURA doesn't just remember the last few messages, it retrieves the most *relevant* past context even from 100 conversations ago.

---

## 🔧 Writing a Skill

Skills are the fastest way to extend AURA. Drop a folder into `skills/` and it auto-loads:

```
skills/
└── my_skill/
    ├── skill.py     ← required
    └── SKILL.md     ← recommended
```

Minimal `skill.py`:
```python
NAME     = "My Skill"
ICON     = "🚀"
KEYWORDS = ["trigger word", "another phrase"]

def run(prompt: str, context: str = "") -> dict:
    return {"success": True, "output": "My response"}
```

See `skills/HOW_TO_WRITE_A_SKILL.md` for the full guide.

---

## 📱 Telegram Setup

1. Message `@BotFather` on Telegram → `/newbot` → copy the token
2. Get your Telegram user ID from `@userinfobot`
3. Add to `.env`:
```env
TELEGRAM_TOKEN=1234567890:ABCdef...
TELEGRAM_ALLOWED_IDS=987654321
```
4. Set `enable_telegram_bot: true` in `aura_config.json`

Available commands: `/skills`, `/schedule`, `/memory`, `/status`, `/help`

---

## 🛠️ Troubleshooting

**Ollama connection error**
Make sure `ollama serve` is running. Check it's accessible at `http://localhost:11434`.

**"Vosk model not found"**
Update `VOSK_MODEL_PATH` in your `.env` to point to the extracted model folder (must contain `conf/` or `am/`).

**"Piper not found" / No voice output**
- Windows: Update `PIPER_PATH` and `PIPER_MODEL` in `.env`
- Linux: Run `bash setup_unix.sh` — it downloads and configures Piper automatically
- macOS: `brew install espeak` as a fallback, or use `setup_unix.sh`

**Playwright not working**
```bash
pip install playwright
playwright install chromium
```

**APScheduler not found**
```bash
pip install apscheduler
```

**Telegram bot not responding**
- Check `TELEGRAM_TOKEN` is set in `.env`
- Check `enable_telegram_bot: true` in `aura_config.json`
- Ensure your user ID is in `TELEGRAM_ALLOWED_IDS`

**Image generation is very slow**
SDXL-Turbo on CPU takes 20-60 seconds on first run. A GPU (CUDA) will make this 10x faster — set `torch_dtype=torch.float16` and `pipe.to("cuda")` in `image_gen.py`.

**Security Agent shows "virtual shell"**
WSL is not installed. Run `wsl --install` (Admin terminal, requires restart). Basic network commands still work via AURA's built-in Python emulator without WSL.

**`webrtcvad` install fails on Windows**
```bash
pip install webrtcvad-wheels
```

**`pyaudio` install fails on Windows**
```bash
pip install pipwin
pipwin install pyaudio
```

**`pyaudio` install fails on Linux**
```bash
sudo apt install portaudio19-dev
pip install pyaudio
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `vosk` | Offline speech recognition |
| `pyaudio` | Microphone audio capture |
| `webrtcvad-wheels` | Voice Activity Detection |
| `pygame` | Audio playback |
| `opencv-python` | Webcam frame capture |
| `pillow` | Image processing |
| `sentence-transformers` | Memory embeddings for semantic search |
| `diffusers` + `torch` | SDXL-Turbo image generation |
| `duckduckgo-search` | Web search |
| `requests` | Ollama API calls |
| `numpy` | Embedding math |
| `psutil` | Process management |
| `pyautogui` | Mouse/keyboard automation |
| `python-dotenv` | `.env` file loading |
| `keyboard` | Hotkey detection |
| `python-docx` | Word document generation |
| `ultralytics` | YOLO real-time object detection |
| `sounddevice` | Music recognition audio capture |
| `playwright` | Browser automation *(new)* |
| `APScheduler` | Proactive scheduler daemon *(new)* |
| `python-telegram-bot` | Telegram remote interface *(new)* |

---

## 🔒 Privacy & Security

- **100% local** — AURA never sends your conversations, voice data, or files to any external server
- All AI inference runs through your local Ollama instance
- Web search uses DuckDuckGo (no tracking)
- Music recognition uses ACRCloud (audio fingerprint only, not raw audio)
- Telegram bot is locked to your user ID — set `TELEGRAM_ALLOWED_IDS` in `.env`
- The Security Agent requires explicit permission before installing any tool
- `self_improvement.py` always backs up files before modifying and validates syntax before overwriting

---

## 🗺️ Roadmap

- [x] Skills plugin system
- [x] Telegram remote interface
- [x] Playwright browser automation
- [x] Proactive scheduler daemon
- [x] Linux / macOS support
- [ ] Wake word detection ("Hey AURA")
- [ ] Multi-monitor support for computer use
- [ ] Multi-agent collaboration (agents spawning sub-agents)
- [ ] Skills marketplace / community registry
- [ ] GPU-accelerated image generation config in GUI
- [ ] Mobile companion app

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

**Want to contribute a skill?** See `skills/HOW_TO_WRITE_A_SKILL.md`.

---

## 📄 License

Custom License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Channing Roe**

---

<div align="center">
  <sub>Built with Python, Ollama, and a lot of late nights.</sub>
</div>
