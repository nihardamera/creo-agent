#!/usr/bin/env bash
# Records docs/demo/creo-demo.mp4 from a fresh compose stack. Needs Docker and uv.
#   bash scripts/demo/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
WORK=$(mktemp -d)
cp scripts/demo/record.mjs "$WORK/"
mkdir -p var docs/demo

docker compose down -v >/dev/null 2>&1 || true
docker compose up -d --build --wait >/dev/null

# The recorder writes a file when it wants the next alert sent.
send() {
  until [ -f "$WORK/$1" ]; do sleep 0.5; done
  shift
  docker compose run --rm replayer python -m creo.replayer "$@" --timeout 180 >/dev/null 2>&1 &
}

docker run --rm --network creo_default -v "$WORK:/work" -w /work \
  mcr.microsoft.com/playwright:v1.64.1-noble \
  sh -c "npm init -y >/dev/null && npm i -s playwright@1.64.1 >/dev/null 2>&1 && node record.mjs" &
REC=$!
send send-fix --split dev --cases c00281 --report /data/demo-fix.json
send send-attack --split adversarial_dev --cases c01167 --report /data/demo-attack.json
wait "$REC"

FFMPEG=$(uv run --quiet --with imageio-ffmpeg==0.6.0 python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FFMPEG" -y -loglevel error -i "$WORK"/video/*.webm -c:v libx264 -preset slow -crf 26 \
  -pix_fmt yuv420p -movflags +faststart docs/demo/creo-demo.mp4
echo "wrote docs/demo/creo-demo.mp4 ($(du -h docs/demo/creo-demo.mp4 | cut -f1))"
