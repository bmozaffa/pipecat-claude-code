#!/bin/bash

# Read prompt from stdin
if [ -n "$1" ]; then
    prompt="$1"
else
    prompt=$(cat)
fi

# Run claude, capture output to temp file to avoid SSH stdout issues
tmpfile=$(mktemp /tmp/claude-out.XXXXXX.json)
echo "$prompt" | claude -p --output-format json --verbose --permission-mode bypassPermissions > "$tmpfile" 2>/dev/null

# Return clean JSON only
cat "$tmpfile"
rm -f "$tmpfile"

