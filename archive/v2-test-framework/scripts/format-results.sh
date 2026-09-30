#!/bin/bash
# format-results.sh - Extract and format test results from JSON files
# Converts escaped \n in stdout/stderr to actual newlines for readability

set -e

usage() {
    echo "Usage: $0 <json-file> [output-file]"
    echo ""
    echo "Extracts test results from JSON and formats stdout/stderr with actual newlines."
    echo "If output-file is not specified, prints to stdout."
    echo ""
    echo "Examples:"
    echo "  $0 build-results.json"
    echo "  $0 build-results.json build-results.txt"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

if [ ! -f "$1" ]; then
    echo "Error: File '$1' not found"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# jq filter to format the results
JQ_FILTER='
.results[] |
"================================================================================",
"TEST: \(.testId // .test_id) - \(.name)",
"Suite: \(.suite) | Pass: \(.pass) | Duration: \(.duration)ms",
"Reason: \(.reason // "N/A")",
"================================================================================",
(.steps // [] | .[] |
  "",
  "--- Step: \(.name) ---",
  "Command: \(.command)",
  "Exit Code: \(.exitCode) | Pass: \(.pass) | Duration: \(.duration)ms",
  (if .stdout and .stdout != "" then "STDOUT:\n\(.stdout)" else empty end),
  (if .stderr and .stderr != "" then "STDERR:\n\(.stderr)" else empty end)
),
""
'

if [ -n "$OUTPUT_FILE" ]; then
    jq -r "$JQ_FILTER" "$INPUT_FILE" > "$OUTPUT_FILE"
    echo "Formatted output written to: $OUTPUT_FILE"
else
    jq -r "$JQ_FILTER" "$INPUT_FILE"
fi
