#!/bin/bash
# ─────────────────────────────────────────────
#  Watcher: monitors Obsidian blog folder and
#  triggers auto-publish on any .md change
# ─────────────────────────────────────────────

OBSIDIAN_BLOG="$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/blog"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PUBLISH_SCRIPT="$SCRIPT_DIR/publish.sh"
LOG_FILE="$SCRIPT_DIR/watcher.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Blog watcher started. Monitoring: $OBSIDIAN_BLOG"

# Debounce: wait 5 seconds after last change before publishing
# This prevents multiple rapid publishes when saving frequently
DEBOUNCE=5

fswatch -0 -e ".*" -i "\\.md$" "$OBSIDIAN_BLOG" | while read -d "" event; do
  log "Change detected: $event"
  
  # Debounce — wait for edits to settle
  sleep $DEBOUNCE
  
  # Drain any queued events during the debounce window
  while read -d "" -t 1 extra_event; do
    log "Additional change: $extra_event (batching)"
  done
  
  log "Running publish..."
  bash "$PUBLISH_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
  log "Waiting for next change..."
done
