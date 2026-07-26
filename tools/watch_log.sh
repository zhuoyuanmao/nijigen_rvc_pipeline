#!/bin/bash
# Readable tail of a pipeline/training log.
#
# The separator models emit tqdm progress bars full of carriage returns, so a
# plain `tail -f` renders as garbage. This strips CRs and drops progress-bar
# lines, leaving only the meaningful events.
#
#   bash tools/watch_log.sh                       # follow the training log
#   bash tools/watch_log.sh /tmp/prep_corpus.log  # follow any log
#   bash tools/watch_log.sh <file> once           # print tail and exit
set -u
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${1:-$PROJECT/characters/otoya_sho_mix/logs/train_v2.log}"
MODE="${2:-follow}"

clean() { tr -d '\r' | grep -a -v -e 'it/s]' -e 'it]' -e '^ *$'; }

if [ ! -f "$LOG" ]; then
    echo "waiting for $LOG ..."
    while [ ! -f "$LOG" ]; do sleep 2; done
fi

if [ "$MODE" = "once" ]; then
    tail -n 400 "$LOG" | clean | tail -n 25
else
    echo "=== following $LOG (Ctrl-C to stop) ==="
    tail -n 200 "$LOG" | clean | tail -n 15
    tail -f -n 0 "$LOG" | clean
fi
