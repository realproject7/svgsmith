#!/usr/bin/env bash
# svgsmith QuadWork batch monitor — exits (waking the PO) only on a real decision point:
#   DONE  = all target issues closed
#   STALL = no progress AND no chat activity for STALL_SECS
#   CAP   = safety cap elapsed
# Otherwise polls quietly. PO relaunches it per phase.
set -uo pipefail
REPO=realproject7/svgsmith
PROJ=svgsmith
QW="$HOME/.local/bin/qw-op"
TARGETS="${TARGETS:-2 3 4 5 6 7 8 9}"
POLL=300
STALL_SECS=3600
CAP_SECS=25200   # 7h
start=$(date +%s)
last_activity=$start
last_closed=-1
last_seq=-1

closed_count() {
  local n=0
  for i in $TARGETS; do
    s=$(gh issue view "$i" --repo "$REPO" --json state --jq .state 2>/dev/null)
    [ "$s" = "CLOSED" ] && n=$((n+1))
  done
  echo "$n"
}
latest_seq() {
  "$QW" read_chat "{\"project\":\"$PROJ\",\"limit\":1}" 2>/dev/null \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[-1]['seq'] if isinstance(d,list) and d else (d.get('seq',-1) if isinstance(d,dict) else -1))" 2>/dev/null || echo -1
}

total=$(echo $TARGETS | wc -w | tr -d ' ')
while :; do
  now=$(date +%s)
  cc=$(closed_count)
  seq=$(latest_seq)
  if [ "$cc" != "$last_closed" ] || [ "$seq" != "$last_seq" ]; then
    last_activity=$now
    last_closed=$cc
    last_seq=$seq
  fi
  if [ "$cc" -ge "$total" ]; then
    echo "DONE: all $total target issues closed ($TARGETS). last_chat_seq=$seq"
    exit 0
  fi
  if [ $((now - start)) -ge $CAP_SECS ]; then
    echo "CAP: ${CAP_SECS}s elapsed. closed=$cc/$total last_chat_seq=$seq"
    exit 0
  fi
  if [ $((now - last_activity)) -ge $STALL_SECS ]; then
    echo "STALL: no issue/chat change for ${STALL_SECS}s. closed=$cc/$total last_chat_seq=$seq"
    exit 0
  fi
  sleep $POLL
done
