#!/usr/bin/env bash
# Executes only after explicit host installation approval. Run from reviewed repository checkout.
set -euo pipefail
umask 077
[ "$(id -u)" = 0 ] || { echo 'run as root'; exit 1; }
[ "$#" = 2 ] || { echo 'usage: install.sh reviewed-host.json full-tool-commit'; exit 1; }
config=$(realpath "$1")
: "${GCP_CLI_PACKAGE_VERSION:?set reviewed exact google-cloud-cli apt package version}"
: "${OPS_AGENT_VERSION:?set reviewed exact Ops Agent version}"
revision=$2
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || exit 1
source_root=$(cd "$(dirname "$0")/../.." && pwd)
[ "$(git -C "$source_root" rev-parse HEAD)" = "$revision" ] || { echo 'tool checkout revision mismatch'; exit 1; }
[ -z "$(git -C "$source_root" status --porcelain --untracked-files=no)" ] || { echo 'tool tracked files changed'; exit 1; }
. /etc/os-release
[ "$ID" = ubuntu ] && [ "$VERSION_ID" = 24.04 ] && [ "$(dpkg --print-architecture)" = amd64 ] || exit 1
python3 - "$config" "$source_root" <<'PY'
import json,sys
sys.path.insert(0, sys.argv[2]+'/deploy/gcp')
from manifest import validate_host
validate_host(json.load(open(sys.argv[1])))
PY
# Versions come from configured signed Ubuntu repositories; record dpkg-query in handoff evidence.
apt-get update
apt-get install -y ca-certificates curl gnupg nginx docker.io docker-compose-v2 openssl python3 iptables
# Official bootstrap bytes verified 2026-10-30. Drift requires a reviewed source/hash update.
staging=$(mktemp -d)
trap 'rm -r "$staging"' EXIT
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg -o "$staging/key.gpg"
printf '%s  %s\n' 3ecc63922b7795eb23fdc449ff9396f9114cb3cf186d6f5b53ad4cc3ebfbb11f "$staging/key.gpg" | sha256sum -c -
gpg --dearmor --output "$staging/keyring.gpg" "$staging/key.gpg"
install -m 0644 "$staging/keyring.gpg" /usr/share/keyrings/capstone-google-cloud.gpg
printf '%s\n' 'deb [signed-by=/usr/share/keyrings/capstone-google-cloud.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/capstone-google-cloud.list
apt-get update
apt-get install -y "google-cloud-cli=$GCP_CLI_PACKAGE_VERSION"
curl -fsSL https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh -o "$staging/ops-agent.sh"
printf '%s  %s\n' d8c492ff1a37036694ae3cd57ac31d241209a0b31c42ec6c6c713b4a97bcfd35 "$staging/ops-agent.sh" | sha256sum -c -
bash "$staging/ops-agent.sh" --also-install --version "$OPS_AGENT_VERSION"
install -d -m 0700 /etc/capstone-ai /var/lib/capstone-ai /var/lib/capstone-ai/releases /var/lib/capstone-ai/operations
install -d -m 0755 /opt/capstone-ai /var/lib/capstone-ai/control
install -m 0600 "$config" /etc/capstone-ai/host.json
for file in manifest.py release.py host_driver.py; do install -m 0644 "$source_root/deploy/gcp/$file" "/opt/capstone-ai/$file"; done
install -m 0755 "$source_root/deploy/gcp/metadata-guard.sh" /opt/capstone-ai/metadata-guard.sh
printf '%s\n' "$revision" > /opt/capstone-ai/REVISION
install -m 0644 "$source_root/deploy/gcp/capstone-ai.service" "$source_root/deploy/gcp/capstone-ai-metadata.service" /etc/systemd/system/
install -m 0600 "$source_root/deploy/gcp/ops-agent.yaml" /etc/google-cloud-ops-agent/config.yaml
systemctl daemon-reload
systemctl enable --now docker nginx capstone-ai-metadata
systemctl enable capstone-ai
systemctl restart google-cloud-ops-agent
echo 'Installed tools; no AI release started. Submit an approved manifest separately.'
