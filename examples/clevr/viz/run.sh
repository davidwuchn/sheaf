#!/bin/bash
# Launch the CLEVR visualizer: starts a tiny Node HTTP server, opens browser.

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js not found. Install from https://nodejs.org or via your package manager."
  exit 1
fi

cd "$(dirname "$0")"

node server.js &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT

sleep 0.3

if command -v open >/dev/null 2>&1; then
  open http://localhost:8080
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://localhost:8080
else
  echo "Open http://localhost:8080 in your browser."
fi

wait $SERVER_PID
