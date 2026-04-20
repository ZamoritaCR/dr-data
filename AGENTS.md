# AGENTS.md — Dr. Data Collaboration Contract

**Read before every session. Binding on you and on Codex/Claude.**

Johan Zamora is the sole approver. If Codex suggests something that contradicts this file, this file wins.

## This repo: `ZamoritaCR/dr-data`

Legacy dr-data. Johan primary. Liqun read + PR only. Contains historical work on feature branches.

## Who does what
- **Johan** — owner, approver, secrets holder
- **Liqun** — V2 UAT engineer, Dubai
- **Codex/Claude** — draft, refactor, review. Never merge. Never hold secrets.

## Git rules (non-negotiable)
1. Branch names: `feat/<user>-<thing>` or `fix/<user>-<thing>`
2. **Never push to main.** Branch protection blocks it.
3. Open a PR. CODEOWNERS auto-assigns Johan. Merge only after approval.
4. No force-pushes on shared branches.
5. Never commit `.env`, `.env.*`, or any credential. gitleaks CI blocks it.

## Secret rules (non-negotiable)
- Secrets live on ROG at `/home/zamoritacr/.env.*` (mode 600). They never leave the server.
- You will **never receive** `PBI_CLIENT_SECRET`, `OPENAI_API_KEY`, or any Azure credential.
- The UAT app holds them server-side. You click "Publish" in the UI; the service principal authenticates.
- If you accidentally see a secret, tell Johan so he can rotate. Do not paste it anywhere.

## Daily heartbeat (commits are the heartbeat)
- Start of session: open your feature branch, push your first commit
- End of session: push your final commit + update `ACTIVITY_LOG.md`
- No Slack, no email. Git is the log.

## Product rules (V2)
1. **Analyst reviews every DAX before publish.** No auto-publish. Human-in-the-loop gate is sacred.
2. **TMDL indentation: tabs, not spaces.**
3. **UAT workspace only**: `226a11c9-8f9a-4374-b4c6-5e01dafa482d`. Any other workspace = P0 incident.
4. **Stage 7 SemanticModel publish must return `Succeeded`** before downstream runs.
5. No emojis in code, commits, or output.

## When stuck
1. Re-read this file
2. Check `ACTIVITY_LOG.md`
3. Ask Codex or Claude — but verify before committing
4. Open a GitHub Discussion (async) or ping Johan (urgent)

---

_Last updated: 2026-04-20 · Johan Zamora_
