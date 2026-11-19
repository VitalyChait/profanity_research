# profanity_research (MinorSafe)

Latent Representation Engineering and Runtime Activation Steering for Age-Appropriate Conversational Systems (12–18).

## Quick Start

### 1. One-Command Repository Entry Point (`./setup.sh` / `make bootstrap`)

```bash
# Local setup + automatic remote GPU server setup when .remote.env is configured
./setup.sh
# Or via Make:
make bootstrap
```

`setup.sh` supports `--local` (local `.venv` + preflight only), `--remote` (full local + remote server setup), and `--check` (preflight & exclusivity checks only). It installs the git pre-commit secret guard, ensures `.remote.env` is gitignored with `0600` permissions, sets up `uv` and `.venv`, installs an SSH key on the remote server (using OpenSSH `SSH_ASKPASS_REQUIRE=force` if `REMOTE_PASSWORD` is set without `sshpass`), verifies VPN reachability and server exclusivity, syncs code, and provisions the remote GPU environment.

### 2. Automated Remote GPU Execution Over University VPN

Connect to your University VPN, configure your gitignored `.remote.env` (copied from `.remote.env.example` on first `./setup.sh` run), then run:

```bash
./setup.sh                                        # Installs SSH key, checks VPN & exclusivity, syncs & sets up remote GPU env
make remote-check                                 # Verify VPN reachability, idle GPUs, and no other active users
bash scripts/remote.sh run data auto              # Build dataset remotely
bash scripts/remote.sh run main gpu_80gb --detach # Run phase inside tmux with smart ETA logging & GPU telemetry
make remote-status                                # Live ETA status, tmux sessions, GPUs, and log tail
make remote-logs                                  # Stream results/logs/latest.log
make remote-pull                                  # Pull results/ back locally (excludes *.safetensors, *.pt)
```

### 3. Manual Server Activation (On Server or Local)

```bash
source scripts/activate.sh auto    # Loads .remote.env (if present), activates .venv, runs preflight
minorsafe preflight --hardware auto
```

### 4. Local Development & Verification

```bash
make lint      # ruff check + format check
make test      # CPU unit tests
make smoke     # End-to-end CPU smoke pipeline
```

For full details, hardware profiles, and troubleshooting, see [docs/RUNBOOK.md](docs/RUNBOOK.md) and [plan.md](plan.md).
