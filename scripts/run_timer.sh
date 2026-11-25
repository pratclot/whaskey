#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SSH_PID=""
cleanup() {
    if [[ -n "$SSH_PID" ]] && kill -0 "$SSH_PID" 2>/dev/null; then
        kill "$SSH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

is_proxy_up() {
    .venv/bin/python -c "import socket; s=socket.create_connection(('127.0.0.1', 1080), 1); s.close()" 2>/dev/null
}

if ! is_proxy_up; then
    echo "🔌 Starting SOCKS5 tunnel to user@proxy-host on 127.0.0.1:1080 ..."
    ssh -N -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -o BatchMode=yes -D 1080 user@proxy-host &
    SSH_PID=$!
    for _ in {1..10}; do
        if is_proxy_up; then
            echo "   ✅ SOCKS5 tunnel ready"
            break
        fi
        if ! kill -0 "$SSH_PID" 2>/dev/null; then
            echo "   ⚠️ SOCKS5 tunnel failed to start; continuing direct"
            SSH_PID=""
            break
        fi
        sleep 0.5
    done
fi

.venv/bin/python -u scripts/tracker.py \
    --refresh-catalog \
    --pending \
    --limit 25 \
    --delay 5 \
    --model gemini-3.5-flash-lite \
    --no-whisper
rc=$?

if [[ "$rc" -ne 0 && "$rc" -ne 2 ]]; then
    echo "❌ tracker.py exited with code $rc" >&2
    exit "$rc"
fi

if [[ -n "$(git status --porcelain src/content/reviews data/channel_videos.jsonl data/transcripts data/runs.jsonl)" ]]; then
    echo "🏗️ Running Astro build ..."
    XDG_CONFIG_HOME=/tmp ASTRO_TELEMETRY_DISABLED=1 /usr/bin/node ./node_modules/.bin/astro build || exit 1

    git add src/content/reviews data/channel_videos.jsonl data/transcripts data/runs.jsonl
    if ! git diff --cached --quiet; then
        git commit -m "chore(timer): process batch"
        git push origin main
    fi
else
    echo "ℹ️ No new changes to commit."
fi

exit "$rc"
