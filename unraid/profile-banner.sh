# shellcheck shell=bash
# Shown when you open a shell in the container (Unraid's console button).
if [ -t 1 ]; then
    printf '\n  TextyMcSpeechy \342\200\224 Piper TTS voice training\n\n'
    printf '  Run  tms          for the command list\n'
    printf '  Run  tms doctor   to check the GPU is working\n\n'
    printf '  Your data lives in /app/tts_dojo (mapped to appdata).\n\n'
fi
