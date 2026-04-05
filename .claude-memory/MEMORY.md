# Memory

## Hard Rules (Johan)

- **NO EMOJIS** — Never use emojis in any output: code, comments, UI text, terminal output, documentation, conversation. Zero exceptions.

## Projects

- **JOAO** (`~/joao-spine/`, `~/joao-interface/`) — AI Exocortex. 16 Council agents. Live at `https://joao.theartofthepossible.io/joao/app`. Railway: `joao-spine-production.up.railway.app`.
- **FocusFlow** (`~/focusflow/`) — ADHD lecture summarizer. Live at `https://focusflow.theartofthepossible.io`.
- **Dr. Data** (`~/projects/dr-data/`) — Tableau-to-PowerBI migration. Streamlit. Port 8502.
- **TAOP Dashboard** — Port 8501.
- **PBIX Extractor** (`~/projects/pbix-extractor/`) — CLI for SQL/DAX/M from .pbix. Production v1.0.
- **joao-spine** (`~/projects/joao-spine/`) — Shared `.env` with API keys.

## Infrastructure

- See [infrastructure.md](infrastructure.md) for full domain/SSL/proxy/agent details.
- Domain: `theartofthepossible.io` (GreenGeeks) -> 70.57.15.252
- Reverse proxy: Traefik via Coolify (auto-SSL, Let's Encrypt)
- Sudo password: available when needed for system operations.

## Council Agent Architecture (v3 -- March 2026)

- **Hot pool** (3 persistent): MAX, CORE, BYTE -- always running in tmux with Claude `--dangerously-skip-permissions`
- **On-demand pool** (12): ARIA, CJ, SOFIA, DEX, GEMMA, LEX, NOVA, SAGE, FLUX, APEX, IRIS, VOLT -- launched by dispatch via file-based launcher when needed
- **Service**: SCOUT -- systemd service, not tmux
- Watchdog (`council_watchdog.sh`) only monitors hot pool + infrastructure services
- Dispatch (`joao_local_dispatch.py` v3) auto-detects if Claude is running; if not, uses `launch_agent.sh` (one-shot `claude --print`)
- Agent detection: process tree inspection (pgrep -P pane_pid), NOT terminal buffer text
- Cloudflare tunnel `joao-dispatch`: dispatch.theartofthepossible.io -> :8100, drdata.theartofthepossible.io -> :8502
- tmux sessions MUST use bash shell (not default sh/dash): `tmux new-session ... "bash"`
- All cron scripts use `flock` for mutual exclusion + `XDG_RUNTIME_DIR`/`DBUS_SESSION_BUS_ADDRESS` for systemctl --user
- JOAO_MASTER_CONTEXT.md: auto-rotated at 50MB by context_watcher.sh

## Environment

- Machine IP: 192.168.0.55 (ROG Strix, Council Server)
- Public IP: 70.57.15.252
- Python venv: `~/taop-agents-env/`
- User: zamoritacr (Johan)
