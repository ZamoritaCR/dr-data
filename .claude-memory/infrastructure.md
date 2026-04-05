# Infrastructure - theartofthepossible.io

## Domains (all via Cloudflare tunnel `joao-dispatch`)
- `joao.theartofthepossible.io` -> :7778 (spine, hub, MCP, chat, council tools)
- `dispatch.theartofthepossible.io` -> :8100 (council dispatch API)
- `drdata.theartofthepossible.io` -> :8502 (Dr. Data Streamlit)
- `focusflow.theartofthepossible.io` -> :8001 (FocusFlow)
- SSL: automatic via Cloudflare tunnel
- GreenGeeks: main site theartofthepossible.io (static pages, FTP deploy)

## Reverse Proxy
- Cloudflare named tunnel `joao-dispatch` (primary -- all subdomains)
- Config: `/etc/cloudflared/config.yml`
- Credentials: `~/.cloudflared/ec7e62bd-*.json`
- Traefik 3.6 via Coolify (legacy, still running)

## Port Forwards
- UPnP: ports 80 + 443 TCP -> 192.168.0.55
- Auto-renewed every 5 min via cron: `~/bin/joao-keepalive.sh`

## Services (systemd --user)
- `joao-dispatch.service` -- Council dispatch API on port 8100
- `joao-spine-local.service` -- Spine FastAPI on port 7778
- `council-scout.service` -- SCOUT Intel Scanner (24/7)
- `joao-tunnel.service` + `joao-tunnel-spine.service` -- Cloudflare tunnels
- `drdata.service` -- Dr. Data Streamlit on port 8502
- `focusflow.service` -- FocusFlow on port 8001
- `browser-agent.service`, `joao-os-agent.service` -- Auxiliary

## Services (system-level)
- `cloudflared.service` -- Main Cloudflare tunnel daemon

## Council Architecture (v3)
- **Hot pool**: MAX, CORE, BYTE (persistent Claude sessions, watchdog-maintained)
- **On-demand**: 12 others (dispatch launches via file-based launcher when needed)
- **Service**: SCOUT (systemd, not tmux)
- Dispatch: `~/joao-spine/joao_local_dispatch.py` v3.0.0
- Watchdog: `~/council/bin/council_watchdog.sh` (hot pool only)
- Launcher: `~/council/bin/launch_agent.sh` (one-shot claude --print)
- Restart: `~/council/restart_agents.sh` (default=hot, `all` flag for all 15)

## Cron Jobs
- `*/5` joao-keepalive.sh -- UPnP renewal + dispatch health
- `*/5` joao_health_monitor.sh -- HTTP checks on all services
- `*/5` council_watchdog.sh -- hot pool agent restart
- `*/5` greengeeks_monitor.py -- disk/domain monitoring
- `*/1` radar_watch.sh -- RADAR_SCORES.json change detection (single-shot, not loop)
- All scripts use flock locks + XDG_RUNTIME_DIR for D-Bus

## Railway
- `JOAO_LOCAL_DISPATCH_URL=https://dispatch.theartofthepossible.io`
- Auto-deploys from `ZamoritaCR/joao-spine` main branch
- Railway URL: `https://joao-spine-production.up.railway.app`

## Key Lessons (March 2026)
- radar_watch.sh was an infinite loop run by cron every minute = fork bomb (9,084 instances)
- Agent detection must use process tree (pgrep -P), not terminal buffer text (persists after death)
- tmux sessions default to /bin/sh (dash) on this system -- must explicitly use bash
- systemctl --user from cron requires XDG_RUNTIME_DIR + DBUS_SESSION_BUS_ADDRESS exports
- JOAO_MASTER_CONTEXT.md grew to 2.2GB from unbounded append -- now auto-rotated at 50MB
- 15 Claude instances = ~5.8GB RAM, unsuitable for 14GB machine -- hot pool (3) saves ~4.6GB

## Hairpin NAT
- Router doesn't support hairpin NAT
- Local access requires /etc/hosts: `192.168.0.55 joao.theartofthepossible.io dispatch.theartofthepossible.io drdata.theartofthepossible.io focusflow.theartofthepossible.io`

## Ollama
- Port 11434, models: phi4 (14.7B), llama3.1 (8B)
