#!/usr/bin/env bash
# Exercises unraid/docker-shim with the exact call shapes the dojo scripts use.
#
# Every case below is a real line from upstream's scripts. If one of these
# regresses, training breaks somewhere deep inside a 15GB image, so this runs in
# CI before the image is built.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shim="$here/../docker-shim"
[ -x "$shim" ] || { echo "shim not executable: $shim" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$shim" "$tmp/docker"
cat > "$tmp/echoargs" <<'ARGS'
#!/usr/bin/env bash
printf 'ARGS:'; for a in "$@"; do printf '[%s]' "$a"; done; printf '\n'
ARGS
chmod +x "$tmp/docker" "$tmp/echoargs"
export PATH="$tmp:$PATH"

pass=0; fail=0
check() {
  if [ "$2" = "$3" ]; then
    pass=$((pass + 1)); printf '  ok   %s\n' "$1"
  else
    fail=$((fail + 1))
    printf '  FAIL %s\n       expected: %s\n       actual:   %s\n' "$1" "$2" "$3"
  fi
}

# scripts/utils/piper_training.sh
check "exec runs the command locally" "ARGS:[/app/x/piper_fit.py][/app/x/fit_params.json]" \
  "$(docker exec textymcspeechy-piper echoargs /app/x/piper_fit.py /app/x/fit_params.json 2>/dev/null)"

# scripts/utils/_tmux_piper_export.sh -- `bash -c "..."` must stay one argv element
check "exec bash -c preserves quoting" "cd /app/piper && export --checkpoint a.ckpt" \
  "$(docker exec textymcspeechy-piper bash -c 'echo "cd /app/piper && export --checkpoint a.ckpt"' 2>/dev/null)"

# scripts/utils/run_tensorboard_server.sh
check "exec -it strips flags, keeps args" "ARGS:[--logdir][/app/tts_dojo/v/logs][--bind_all]" \
  "$(docker exec -it textymcspeechy-piper echoargs --logdir /app/tts_dojo/v/logs --bind_all 2>/dev/null)"

# scripts/utils/checkpoint_grabber.sh
check "exec ps aux succeeds" "yes" \
  "$(docker exec textymcspeechy-piper ps aux >/dev/null 2>&1 && echo yes || echo no)"

# tts_dojo/ESPEAK_RULES/apply_custom_rules.sh
check "exec -u root -it still runs" "ARGS:[en]" \
  "$(docker exec -u root -it textymcspeechy-piper echoargs en 2>/dev/null)"

# scripts/utils/_control_console.sh, run_training.sh, stop_container.sh
# These MUST NOT take the container down -- it is hosting the dojo.
check "stop is a no-op" "0" "$(docker stop textymcspeechy-piper >/dev/null 2>&1; echo $?)"
check "container stop is a no-op" "0" "$(docker container stop textymcspeechy-piper >/dev/null 2>&1; echo $?)"

# prebuilt_container_run.sh probes for an nvidia runtime before starting
check "info reports an nvidia runtime" "yes" \
  "$(docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia && echo yes || echo no)"

check "inspect reports running" "true" \
  "$(docker inspect -f '{{.State.Running}}' textymcspeechy-piper 2>/dev/null)"

check "exec -w changes directory" "$tmp" \
  "$(docker exec -w "$tmp" textymcspeechy-piper pwd 2>/dev/null)"

# A missing workdir must fail loudly rather than run the command elsewhere.
check "exec -w on a missing dir fails" "1" \
  "$(docker exec -w /no/such/dir textymcspeechy-piper true >/dev/null 2>&1; echo $?)"

# Only this container's name may map to local execution.
check "exec into another container refuses" "1" \
  "$(docker exec some-other-container true >/dev/null 2>&1; echo $?)"

# local_container_run.sh
check "compose up is a no-op" "0" "$(docker compose up -d >/dev/null 2>&1; echo $?)"

printf '\n  %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
