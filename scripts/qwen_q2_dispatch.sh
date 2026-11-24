#!/usr/bin/env bash
# Submit once per receipt directory. Recover a lost response with GET requests.
set -euo pipefail
repo=jasonjiang8866/waferVLM
api="repos/$repo/actions/workflows/qwen-q2.yml"
header="X-GitHub-Api-Version: 2026-03-10"

usage() {
  echo "Usage: bash $0 submit RECEIPT_DIR REF REVIEWED_SHA [--paid] | find RECEIPT_DIR" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
mode=$1
dir=$2
case "$mode" in
  submit)
    [[ $# == 4 || ( $# == 5 && $5 == --paid ) ]] || usage
    ref=$3
    sha=$4
    [[ "$sha" =~ ^[0-9a-f]{40}$ && -n "$ref" ]] || usage
    paid=false
    if [[ $# == 5 ]]; then
      [[ "$ref" == main ]] || usage
      paid=true
    fi
    gh auth status --hostname github.com >/dev/null
    payload=$(jq -n --arg ref "$ref" --arg sha "$sha" --arg id "$(python3 -c 'import uuid; print(uuid.uuid4().hex)')" \
      --argjson paid "$paid" '{ref:$ref,inputs:{request_id:$id, expected_sha:$sha, confirm_paid:$paid}}')
    mkdir -- "$dir" || { echo "Choose find to inspect an existing request." >&2; exit 1; }
    printf '%s\n' "$payload" > "$dir/request.json"
    echo "Request saved in $dir/request.json before dispatch." >&2
    gh api --hostname github.com -H "$header" -X POST "$api/dispatches" --input "$dir/request.json" > "$dir/response.json" || {
      echo "Dispatch outcome unknown. Use find with this receipt directory." >&2
      exit 1
    }
    jq -es --arg base "https://github.com/$repo/actions/runs/" '
      .[0] | select((.workflow_run_id | type) == "number" and .workflow_run_id > 0)
      | select(.html_url == ($base + (.workflow_run_id | tostring))) | .html_url
    ' -r "$dir/response.json" || { echo "Missing run receipt. Use find with this directory." >&2; exit 1; }
    ;;
  find)
    [[ $# == 2 ]] || usage
    input="$dir/request.json"
    ref=$(jq -er '.ref' "$input")
    sha=$(jq -er '.inputs.expected_sha' "$input")
    title=$(jq -er '"Qwen Q2 " + (if .inputs.confirm_paid then "paid" else "preflight" end) + " [" + .inputs.request_id + "]"' "$input")
    gh api --hostname github.com -H "$header" --method GET "$api/runs" --paginate \
      -f event=workflow_dispatch -f branch="$ref" -f head_sha="$sha" -F per_page=100 |
      jq -ers --arg title "$title" '[.[].workflow_runs[] | select(.display_title == $title)] |
        if length == 1 then .[0] | {id,html_url,status,conclusion}
        elif length == 0 then error("No matching run yet. Keep the receipt for another read-only lookup.")
        else error("Multiple matching runs. Inspect workflow history before taking action.") end'
    ;;
  *) usage ;;
esac
