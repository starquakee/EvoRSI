#!/usr/bin/env bash
# Initialize private auth + runtime dirs for the rsi-trustworthy stack.
# The auth file and the worker control key live OUTSIDE Git (.runtime/ is
# gitignored), are created with mode 600 and their values are never
# printed. Refuses to overwrite existing credentials (delete the files
# manually to rotate).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)/.runtime/rsi-trustworthy"
AUTH_FILE="${RUNTIME_DIR}/auth.env"
CONTROL_KEY_FILE="${RUNTIME_DIR}/worker_control_key"

mkdir -p "${RUNTIME_DIR}/storage/mlsandbox/jobs" \
         "${RUNTIME_DIR}/storage/mlsandbox/uploads" \
         "${RUNTIME_DIR}/workdir"
chmod 700 "${RUNTIME_DIR}" || true
# The worker runs with cap_drop ALL (plus only SETUID/SETGID/KILL for the
# supervisor), so container root has NO CAP_DAC_OVERRIDE: it can only
# write where permission bits allow it. The per-job scratch must therefore
# be world-writable (public input stays read-only at the mount level).
chmod 777 "${RUNTIME_DIR}/workdir"

umask 077

if [[ -f "${CONTROL_KEY_FILE}" ]]; then
    echo "worker_control_key already exists (not overwritten)." >&2
else
    # Batch-2: authenticates the dispatcher->worker control channel. The
    # worker supervisor reads it once at startup through a privilege-
    # dropped helper child; candidate processes cannot open it.
    python3 -c 'import secrets; print(secrets.token_hex(32))' > "${CONTROL_KEY_FILE}"
    chmod 600 "${CONTROL_KEY_FILE}"
    echo "Created worker control key at ${CONTROL_KEY_FILE} (mode 600, value not shown)." >&2
fi

if [[ -f "${AUTH_FILE}" ]]; then
    echo "auth.env already exists at ${AUTH_FILE} (not overwritten)." >&2
    exit 0
fi

API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
DB_PASS="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"

cat > "${AUTH_FILE}" <<EOF
SANDBOX_API_KEYS=${API_KEY}
POSTGRES_PASSWORD=${DB_PASS}
DB_PASSWORD=${DB_PASS}
EOF
chmod 600 "${AUTH_FILE}"
echo "Created private auth file at ${AUTH_FILE} (mode 600, values not shown)." >&2
