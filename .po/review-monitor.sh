#!/usr/bin/env bash
set -uo pipefail
PROJ=svgsmith; QW="$HOME/.local/bin/qw-op"
POLL=240; CAP_SECS=5400  # 90m cap
start=$(date +%s)
while :; do
  now=$(date +%s)
  hit=$("$QW" read_chat "{\"project\":\"$PROJ\",\"limit\":15}" 2>/dev/null \
    | python3 -c "
import sys,json
d=json.load(sys.stdin); m=d if isinstance(d,list) else d.get('messages',[])
# only count REVIEW COMPLETE from non-user senders (exclude the PO instruction)
print('YES' if any('REVIEW COMPLETE' in str(x.get('text','')) and x.get('sender')!='user' for x in m) else 'NO')
" 2>/dev/null || echo NO)
  if [ "$hit" = "YES" ]; then echo "REVIEW_COMPLETE detected (from team)"; exit 0; fi
  if [ $((now-start)) -ge $CAP_SECS ]; then echo "CAP: ${CAP_SECS}s elapsed, no REVIEW COMPLETE yet"; exit 0; fi
  sleep $POLL
done
