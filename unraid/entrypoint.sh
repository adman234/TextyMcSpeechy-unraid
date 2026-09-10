#!/usr/bin/env bash
# Entrypoint for the all-in-one Unraid image.
#
# Starts as root, sets up the unprivileged user, seeds the persistent dojo on
# first run, then drops privileges and idles. The container is a workbench you
# attach to, not a daemon -- see unraid/README-unraid.md.
#
# Also tolerates being run non-root (upstream's docker-compose.yml sets
# `user:`), in which case the root-only setup is skipped and the command runs
# as-is, so the original compose workflow is unaffected.
set -euo pipefail

PUID="${PUID:-99}"
PGID="${PGID:-100}"
UMASK_SET="${UMASK:-000}"
TMS_USER="${TMS_USER:-tms}"
RUN_AS_ROOT="${RUN_AS_ROOT:-false}"
DOJO_DIR=/app/tts_dojo
SEED_DIR=/opt/tms-seed/tts_dojo

log() { printf '[textymcspeechy] %s\n' "$*"; }

umask "${UMASK_SET}"

if [ "$(id -u)" -ne 0 ]; then
    log "not running as root (uid=$(id -u)); skipping user setup and seeding."
    exec /usr/bin/tini -g -- "$@"
fi

if [ -n "${TZ:-}" ] && [ -f "/usr/share/zoneinfo/${TZ}" ]; then
    ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime
    echo "${TZ}" > /etc/timezone
fi

# --- user / group -----------------------------------------------------------
# Default to Unraid's nobody:users so exported voices look like every other
# file on the array and stay readable over SMB.
if ! getent group "${PGID}" >/dev/null 2>&1; then
    if getent group "${TMS_USER}" >/dev/null 2>&1; then
        groupmod -o -g "${PGID}" "${TMS_USER}"
    else
        groupadd -o -g "${PGID}" "${TMS_USER}"
    fi
fi

if id -u "${TMS_USER}" >/dev/null 2>&1; then
    usermod -o -u "${PUID}" -g "${PGID}" -d "/home/${TMS_USER}" -s /bin/bash "${TMS_USER}" >/dev/null
else
    useradd -o -u "${PUID}" -g "${PGID}" -d "/home/${TMS_USER}" -s /bin/bash -M "${TMS_USER}"
fi
mkdir -p "/home/${TMS_USER}"
chown "${PUID}:${PGID}" "/home/${TMS_USER}"

case "${RUN_AS_ROOT}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) DROP_PRIVS=false ;;
    *) DROP_PRIVS=true ;;
esac

# --- seed the persistent dojo ----------------------------------------------
# /app/tts_dojo is the user's appdata mount and starts empty. The dojo tree is
# not just scripts -- DATASETS, PRETRAINED_CHECKPOINTS and DOJO_CONTENTS are the
# working directories the whole workflow assumes exist, so an empty mount means
# nothing works. Seed once, then never touch it again: after first run this is
# the user's data, and SETTINGS.txt in particular is theirs to edit.
if [ ! -e "${DOJO_DIR}/newdojo.sh" ]; then
    log "first run: seeding ${DOJO_DIR} from the image"
    mkdir -p "${DOJO_DIR}"
    cp -a "${SEED_DIR}/." "${DOJO_DIR}/"
    chown -R "${PUID}:${PGID}" "${DOJO_DIR}"
    log "seeded. Your datasets, dojos and voices live here and survive updates."
else
    log "existing dojo found at ${DOJO_DIR}; leaving it untouched"
    # Claim only the top level -- a recursive chown over hundreds of GB of
    # checkpoints on every start would be slow and pointless.
    if [ "$(stat -c %u "${DOJO_DIR}")" != "${PUID}" ]; then
        log "adjusting ownership of ${DOJO_DIR} to ${PUID}:${PGID}"
        chown "${PUID}:${PGID}" "${DOJO_DIR}"
    fi
fi

# The dojo scripts must be executable. A dojo restored from a backup, copied
# off an SMB share, or seeded onto a filesystem that lost the bits will
# otherwise fail in confusing ways on the first run.
# Depth 5 reaches <voice>_dojo/scripts/utils/, the deepest scripts there are.
find "${DOJO_DIR}" -maxdepth 5 -name '*.sh' -exec chmod +x {} + 2>/dev/null || true

# --- GPU sanity -------------------------------------------------------------
# Cheap to check, and catches the single most common misconfiguration (missing
# --runtime=nvidia) at start rather than 40 minutes into a training run.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    log "GPU: $(nvidia-smi -L | head -n1)"
else
    log "WARNING: no NVIDIA GPU visible. Training will not work."
    log "         Check '--runtime=nvidia' is in Extra Parameters and that the"
    log "         Unraid Nvidia-Driver plugin is installed. Run 'tms doctor'."
fi

if [ "${DROP_PRIVS}" = "true" ]; then
    log "ready as ${TMS_USER} (uid=${PUID} gid=${PGID}). Open the console and run: tms"
    export HOME="/home/${TMS_USER}" USER="${TMS_USER}"
    exec /usr/bin/tini -g -- gosu "${TMS_USER}" "$@"
fi

log "RUN_AS_ROOT is set: running as root; files created will be root-owned"
exec /usr/bin/tini -g -- "$@"
