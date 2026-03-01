#!/bin/bash
set -e

cleanup() {
    echo "Received shutdown signal, stopping application..."
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        kill -TERM "$APP_PID" 2>/dev/null || true
        sleep 30
        if kill -0 "$APP_PID" 2>/dev/null; then
            echo "Application did not stop gracefully, force killing..."
            kill -KILL "$APP_PID" 2>/dev/null || true
        fi
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "=========================================="
echo "Agentic AutoML Platform Starting..."
echo "=========================================="
echo "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
echo "PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}"

mkdir -p /app/data

echo "Initializing job queue database..."
sqlite3 /app/data/jobs.db "CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    result_path TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL
)"

echo "Database initialized at /app/data/jobs.db"
echo "Starting application: $@"
echo "=========================================="

exec "$@" &
APP_PID=$!
wait $APP_PID