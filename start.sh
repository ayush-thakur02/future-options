#!/usr/bin/env bash
# Host a minimal local OAuth callback through the persistent Azure Dev Tunnel.
#
# On first run this creates tunnel port 8765. Copy the HTTPS URL printed by
# `devtunnel host` into both Upstox's Redirect URL field and UPSTOX_REDIRECT_URI
# in .env. Keep the value exactly the same in both places.

set -Eeuo pipefail

TUNNEL_ID="${TUNNEL_ID:-sneaky-hill-03x9hbb.use}"
CALLBACK_PORT="${CALLBACK_PORT:-8765}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v devtunnel >/dev/null 2>&1; then
    echo "devtunnel is not installed or is not on PATH." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable '$PYTHON_BIN' was not found." >&2
    exit 1
fi

# A fresh empty directory ensures the public tunnel can never serve files from
# this repository (especially .env). The callback code is copied from the
# browser address bar and entered into `niftypulse login` manually.
callback_dir="$(mktemp -d)"
callback_pid=""

cleanup() {
    if [[ -n "$callback_pid" ]] && kill -0 "$callback_pid" 2>/dev/null; then
        kill "$callback_pid" 2>/dev/null || true
        wait "$callback_pid" 2>/dev/null || true
    fi
    rmdir "$callback_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if ! devtunnel port show "$TUNNEL_ID" --port-number "$CALLBACK_PORT" >/dev/null 2>&1; then
    echo "Creating port $CALLBACK_PORT on tunnel $TUNNEL_ID ..."
    devtunnel port create "$TUNNEL_ID" --port-number "$CALLBACK_PORT" --protocol http
fi

"$PYTHON_BIN" -m http.server "$CALLBACK_PORT" \
    --bind 127.0.0.1 \
    --directory "$callback_dir" \
    >/dev/null 2>&1 &
callback_pid=$!

# Give Python a moment to report an occupied or unavailable port before
# starting the public tunnel.
sleep 0.2
if ! kill -0 "$callback_pid" 2>/dev/null; then
    echo "Could not start the local callback listener on port $CALLBACK_PORT." >&2
    exit 1
fi

cat <<EOF
Local OAuth callback is listening on 127.0.0.1:$CALLBACK_PORT.
Starting Azure Dev Tunnel $TUNNEL_ID ...

Copy the HTTPS URL printed below into Upstox and .env if you have not already.
Leave this script running while you complete \`niftypulse login\`.
Press Ctrl-C when login has finished; the local listener will be removed.

EOF

# The port is configured above. Passing it again while hosting an existing
# tunnel asks the service to batch-update ports and is rejected.
devtunnel host "$TUNNEL_ID"
