#!/bin/sh
# Runs inside node:22-slim on the compose network. $TOK and $RUN come from the environment.
I="npx -y @modelcontextprotocol/inspector@2.11.0 --cli http://mcp:8003/mcp --transport http"
echo "== tools/list (creo:read)"
$I --header "Authorization: Bearer $TOK" --method tools/list
echo "== tools/call get_run $RUN"
$I --header "Authorization: Bearer $TOK" --method tools/call --tool-name get_run --tool-arg "run_id=$RUN"
echo "== tools/call request_approval with a read-only token"
$I --header "Authorization: Bearer $TOK" --method tools/call --tool-name request_approval --tool-arg action=restart_iis --tool-arg 'params={"host":"ash-web-01"}' --tool-arg why=test 2>&1; echo "exit=$?"
echo "== tools/list with a bad token"
$I --header "Authorization: Bearer not-a-token" --method tools/list 2>&1 | tail -3; echo "exit=$?"

# Usage, from the repo root with the compose stack up:
#   TOK=$(docker compose exec -T mcp python -m creo.mcp_server token --scopes creo:read --subject inspector | tail -1)
#   RUN=$(curl -s 'http://localhost:8000/api/runs?limit=1' | python3 -c "import json,sys;print(json.load(sys.stdin)[0]['run_id'])")
#   docker run --rm --network creo_default -e TOK="$TOK" -e RUN="$RUN" \
#     -v "$PWD/scripts/mcp_inspector.sh:/inspect.sh:ro" node:22-slim sh /inspect.sh
