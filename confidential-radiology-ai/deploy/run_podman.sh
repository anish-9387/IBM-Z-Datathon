#!/usr/bin/env bash
# Container deployment sketch (NOT executed on our LinuxONE server: podman is not installed there).
# Usage: ./deploy/run_podman.sh <study_id>     (run from the project root, after the notebook created crai_data/)
set -euo pipefail
STUDY="${1:?study id}"
podman build -t crai-broker -f deploy/Containerfile.broker .
podman build -t crai-guest  -f deploy/Containerfile.guest  .
# Private network with no route to the internet: only broker <-> guest traffic.
podman network exists crai-net || podman network create --internal crai-net
# Broker: the only container that mounts the key store.
podman container exists crai-broker || podman run -d --name crai-broker --network crai-net \
  -v "$PWD/crai_data/broker_store:/data/broker_store:Z" -v "$PWD/crai_data/logs:/data/logs:Z" crai-broker
# Guest: read-only root FS, read-only vault + models, no key store, no internet, no capabilities.
podman run --rm --network crai-net --read-only --cap-drop=ALL --security-opt no-new-privileges \
  -v "$PWD/crai_data/vault:/data/vault:ro,Z" -v "$PWD/crai_data/models:/data/models:ro,Z" \
  crai-guest "$STUDY" "/data/models/$(python3 -c 'import json;print(json.load(open("crai_data/broker_store/policy.json"))["model_file"].split("/")[-1])')"
