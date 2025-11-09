#!/usr/bin/env bash
# ==============================================
# Simple Process Manager for Flask App
# ==============================================

set -e  # exit on unhandled error

APP_NAME="server.py"
VENV_DIR="venv"
LOG_DIR="logs"
PYTHON=${PYTHON:-python3}

# Create log directory if missing
mkdir -p "$LOG_DIR"

# ---- Step 3: Run and restart on crash ----
echo "[INFO] Starting $APP_NAME with autorestart..."
echo "[INFO] Logs: $LOG_DIR/server.log and $LOG_DIR/server.err"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server..." | tee -a "$LOG_DIR/server.log"
  python "$APP_NAME" >>"$LOG_DIR/server.log" 2>>"$LOG_DIR/server.err"
  EXIT_CODE=$?

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server exited with code $EXIT_CODE" | tee -a "$LOG_DIR/server.log"

  # Restart delay
  echo "[INFO] Restarting in 5 seconds..."
  sleep 5
done
