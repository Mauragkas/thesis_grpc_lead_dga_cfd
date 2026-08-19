#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Check if at least an input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_file> [output_file]"
    echo "Example: $0 domain-model.mmd domain-model.png"
    exit 1
fi

INPUT="$1"

# If output file is not specified, default to <input-basename>.png
OUTPUT="${2:-${INPUT%.*}.png}"

CONFIG_FILE="puppeteer-config.json"

# Auto-create puppeteer-config.json if missing
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file missing. Generating temporary '$CONFIG_FILE'..."
    cat << 'EOF' > "$CONFIG_FILE"
{
  "executablePath": "/usr/bin/chromium",
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
EOF
fi

echo "Rendering '$INPUT' -> '$OUTPUT'..."

mmdc -i "$INPUT" \
     -o "$OUTPUT" \
     -s 5 \
     -b "#1a1a1a" \
     -t dark \
     -p "$CONFIG_FILE"

echo "Done! Output saved to: $OUTPUT"
