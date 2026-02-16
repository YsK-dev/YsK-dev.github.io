#!/bin/bash
# ─────────────────────────────────────────────
#  Auto-publish: Obsidian → Hugo → GitHub Pages
# ─────────────────────────────────────────────

OBSIDIAN_BLOG="$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/blog"
HUGO_ROOT="$HOME/Desktop/untitled folder/blog-source"
HUGO_POSTS="$HUGO_ROOT/content/post"
HUGO_IMAGES="$HUGO_ROOT/static/img/posts"
LOG_FILE="$HUGO_ROOT/scripts/publish.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ── Convert Obsidian wiki-link images to standard Markdown ──
# Transforms ![[image.png]] → ![image](/img/posts/image.png)
convert_obsidian_images() {
  local text="$1"
  # Match ![[filename.ext]] where ext is an image type
  # First convert to standard markdown syntax
  text=$(echo "$text" | sed -E 's/!\[\[([^]]+\.(png|jpg|jpeg|gif|webp|svg|PNG|JPG|JPEG|GIF|WEBP|SVG))\]\]/![\1](\/img\/posts\/\1)/g')
  # Then URL-encode spaces in the parenthesized URL portion: ](...)  
  # Use perl for the lookahead-based space replacement inside markdown image URLs
  text=$(echo "$text" | perl -pe 's{(!\[[^\]]*\]\()([^)]+)(\))}{my ($pre,$url,$post)=($1,$2,$3); $url=~s/ /%20/g; "$pre$url$post"}ge')
  echo "$text"
}

# ── Sanity checks ──
if [ ! -d "$OBSIDIAN_BLOG" ]; then
  log "ERROR: Obsidian blog folder not found: $OBSIDIAN_BLOG"
  exit 1
fi

if [ ! -d "$HUGO_ROOT" ]; then
  log "ERROR: Hugo root not found: $HUGO_ROOT"
  exit 1
fi

log "Starting publish..."

# ── Ensure image directory exists ──
mkdir -p "$HUGO_IMAGES"

# ── Sync images from Obsidian to Hugo static ──
for file in "$OBSIDIAN_BLOG"/*.{png,jpg,jpeg,gif,webp,svg,PNG,JPG,JPEG}; do
  [ -f "$file" ] || continue
  
  filename=$(basename "$file")
  dest="$HUGO_IMAGES/$filename"
  
  if [ -f "$dest" ]; then
    if diff -q "$file" "$dest" > /dev/null 2>&1; then
      continue
    fi
  fi
  
  cp "$file" "$dest"
  log "Synced image: $filename"
  CHANGED=true
done

# ── Sync markdown files from Obsidian to Hugo ──
# Only copy .md files. Preserve Hugo's existing posts.
CHANGED=false

for file in "$OBSIDIAN_BLOG"/*.md; do
  [ -f "$file" ] || continue
  
  filename=$(basename "$file")
  dest="$HUGO_POSTS/$filename"
  
  # Read the file content
  content=$(cat "$file")
  
  # Check if file has Hugo front matter (starts with ---)
  if ! echo "$content" | head -1 | grep -q '^---'; then
    # No front matter — add default front matter
    # Extract title from filename (remove .md, replace dashes/underscores with spaces)
    title=$(echo "$filename" | sed 's/\.md$//' | sed 's/[-_]/ /g')
    date=$(date '+%Y-%m-%d')
    
    # Convert Obsidian image links to standard Markdown
    content=$(convert_obsidian_images "$content")

    new_content="---
title: \"$title\"
date: $date
type: post
tags: []
draft: false
---

$content"
    
    # Check if destination exists and content is different
    if [ -f "$dest" ]; then
      existing=$(cat "$dest")
      if [ "$new_content" = "$existing" ]; then
        continue
      fi
    fi
    
    echo "$new_content" > "$dest"
    log "Published (added front matter): $filename"
    CHANGED=true
  else
    # Has front matter — convert Obsidian images, then copy
    converted=$(convert_obsidian_images "$content")
    
    if [ -f "$dest" ]; then
      existing=$(cat "$dest")
      if [ "$converted" = "$existing" ]; then
        continue
      fi
    fi
    
    echo "$converted" > "$dest"
    log "Published: $filename"
    CHANGED=true
  fi
done

# ── Check for deleted files ──
# Files in Hugo that came from Obsidian but were removed from Obsidian
# We track synced files in a manifest
MANIFEST="$HUGO_ROOT/scripts/.obsidian-manifest"
touch "$MANIFEST"

# Build current Obsidian file list
CURRENT_FILES=""
for file in "$OBSIDIAN_BLOG"/*.md; do
  [ -f "$file" ] || continue
  CURRENT_FILES="$CURRENT_FILES$(basename "$file")\n"
done

# Check manifest for files that no longer exist in Obsidian
while IFS= read -r tracked; do
  [ -z "$tracked" ] && continue
  if ! echo -e "$CURRENT_FILES" | grep -qx "$tracked"; then
    if [ -f "$HUGO_POSTS/$tracked" ]; then
      rm "$HUGO_POSTS/$tracked"
      log "Removed (deleted from Obsidian): $tracked"
      CHANGED=true
    fi
  fi
done < "$MANIFEST"

# Update manifest
echo -e "$CURRENT_FILES" | sort | uniq | grep -v '^$' > "$MANIFEST"

# ── Build and push if anything changed ──
if [ "$CHANGED" = true ]; then
  cd "$HUGO_ROOT" || exit 1
  
  # Build
  log "Building Hugo site..."
  hugo_output=$(hugo 2>&1)
  if [ $? -ne 0 ]; then
    log "ERROR: Hugo build failed: $hugo_output"
    exit 1
  fi
  log "Hugo build OK"
  
  # Commit and push
  git add -A
  git commit -m "Auto-publish from Obsidian ($(date '+%Y-%m-%d %H:%M'))"
  
  if git push origin main 2>&1; then
    log "Pushed to GitHub successfully"
  else
    log "ERROR: Git push failed"
    exit 1
  fi
else
  log "No changes detected, skipping."
fi

log "Done."
