#!/usr/bin/env bash
# M2 acceptance: kill the agent while a run is waiting, restart it, and check
# the run resumes and finishes on its labelled path.
#   scripts/kill_test.sh            (expects `docker compose up -d` to be running)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p var
CASE=${CASE:-c00309}  # queue-backlog: waits 10 + 5 + 5 scripted minutes, then a fix
docker compose run --rm replayer python -m creo.replayer --split dev --cases "$CASE" \
  --auto-approve --timeout 180 --report /data/kill-test-report.json > var/kill-test.log 2>&1 &
replay=$!
sleep 6
echo "killing the agent mid-wait"; docker compose kill agent
sleep 10
echo "restarting it"; docker compose start agent
wait "$replay"
tail -9 var/kill-test.log
python3 - <<'PY'
import json, sys
s = json.load(open("var/kill-test-report.json"))["summary"]
ok = s["finished"] == s["cases"] == s["path_matches"] == 1
print("PASS: the run resumed and took its labelled path" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
