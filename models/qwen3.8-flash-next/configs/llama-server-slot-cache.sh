#!/bin/bash
# Keep llama-server's prompt cache (slot 0) across restarts.
#
# A restart empties the KV cache, and re-reading a long agent conversation costs minutes (about
# 9.5 min for 166K tokens of Qwen3.8-Flash-Next on four B70s). llama-server can save a slot to
# disk and restore it (start it with --slot-save-path <dir>); the patched build in
# configs/patches/ (0010) also saves the MTP draft's copy next to it (<file>.dft).
#
#   save      save slot 0 unless a hold is set. Run it from the unit's ExecStop=.
#   autosave  save slot 0 once the agent has gone idle (no new task for a whole timer period) and the
#             slot changed since the last save, unless a hold is set. Run it from a timer every few
#             minutes, so a crash, OOM kill or power cut loses only the turns since the last idle spell.
#   restore   restore the saved slot if it was saved by an identical setup (see SIGNATURE below).
#             Run it from ExecStartPost=, after the server answers /health.
#   hold      save now, then keep automatic saves from overwriting it (before tests or benchmarks
#             that would fill the slot with their own prompts).
#   release   drop the hold.
#   status    show what is saved.
#
# Saves under MIN_TOKENS tokens are discarded, so an idle restart cannot replace a useful save. autosave
# also skips a slot under half the size of the last save made in the past 12 h: a side request that
# displaced the main conversation into the host-RAM prompt cache (--cache-ram), which a save cannot reach.
# Never fails the caller (always exits 0). The saved file holds the conversation: keep SLOT_DIR 0700.
#
# Fill in (environment variables; defaults in brackets):
#   SLOT_DIR       the directory given to llama-server as --slot-save-path        [/path/to/slots]
#   SERVER_URL     the server's base URL                                          [http://127.0.0.1:8080]
#   API_KEY_FILE   0600 file holding the API key, if the server needs one; the key is passed to curl
#                  on stdin, never on the command line                            [empty: no key]
#   SIGNATURE      a file that describes the running setup: model, draft, build, context, KV types,
#                  draft mode (one "key=value" per line). Write it from the launch script before
#                  exec'ing llama-server. A saved slot is only restored when this file matches the
#                  copy saved next to the slot.                                   [/run/llama-server/slot-signature]
#   MIN_TOKENS     smallest slot worth saving                                     [2048]
#   UNIT           the server's systemd unit; autosave skips when it is not active, and keys "already
#                  saved" on its InvocationID so a restart starts afresh              [empty: no check]
set -u

DIR=${SLOT_DIR:-/path/to/slots}
URL=${SERVER_URL:-http://127.0.0.1:8080}
SIG_RUN=${SIGNATURE:-/run/llama-server/slot-signature}
KEY_FILE=${API_KEY_FILE:-}
MIN_TOKENS=${MIN_TOKENS:-2048}
UNIT=${UNIT:-}

log() { echo "slot-cache: $*"; }

# POST a slot action; the key goes to curl on stdin, never on the command line
slot_action() {   # action filename timeout
    if [ -n "$KEY_FILE" ] && [ -r "$KEY_FILE" ]; then
        printf 'Authorization: Bearer %s\n' "$(head -n1 "$KEY_FILE")" |
            curl -s -m "$3" -H @- -H 'Content-Type: application/json' \
                 -X POST "$URL/slots/0?action=$1" -d "{\"filename\": \"$2\"}"
    else
        curl -s -m "$3" -H 'Content-Type: application/json' \
             -X POST "$URL/slots/0?action=$1" -d "{\"filename\": \"$2\"}"
    fi
}

# GET /slots; the key goes to curl on stdin
slots_json() {
    if [ -n "$KEY_FILE" ] && [ -r "$KEY_FILE" ]; then
        printf 'Authorization: Bearer %s\n' "$(head -n1 "$KEY_FILE")" | curl -s -m 10 -H @- "$URL/slots"
    else
        curl -s -m 10 "$URL/slots"
    fi
}

json_field() {    # field; reads JSON on stdin
    python3 -c "import json,sys
try: print(json.load(sys.stdin).get('$1', ''))
except Exception: print('')"
}

do_save() {   # returns 1 only when the server could not be asked (down, or another save running)
    local tmp=slot0.tmp.bin out n
    # one save at a time (autosave and the save on stop can overlap)
    exec 9>"$DIR/.lock"
    flock -w 300 9 || { log "another save is still running; skipped"; return 1; }
    rm -f "$DIR/$tmp" "$DIR/$tmp.dft"
    out=$(slot_action save "$tmp" 300) || { log "save request failed (server down?)"; return 1; }
    n=$(printf '%s' "$out" | json_field n_saved)
    if [ -z "$n" ]; then
        log "save failed: $(printf '%s' "$out" | head -c 300)"
        rm -f "$DIR/$tmp" "$DIR/$tmp.dft"
        return 0
    fi
    if [ "$n" -lt "$MIN_TOKENS" ]; then
        log "slot holds $n tokens (< $MIN_TOKENS); keeping the previous save"
        rm -f "$DIR/$tmp" "$DIR/$tmp.dft"
        return 0
    fi
    mv -f "$DIR/$tmp" "$DIR/slot0.bin"
    if [ -f "$DIR/$tmp.dft" ]; then mv -f "$DIR/$tmp.dft" "$DIR/slot0.bin.dft"; else rm -f "$DIR/slot0.bin.dft"; fi
    if [ -r "$SIG_RUN" ]; then cp -f "$SIG_RUN" "$DIR/slot0.sig"; else rm -f "$DIR/slot0.sig"; fi
    printf 'saved_at=%s\nn_tokens=%s\n' "$(date -u +%FT%TZ)" "$n" > "$DIR/slot0.meta"
    log "saved $n tokens ($(du -h "$DIR/slot0.bin" | cut -f1), draft $( [ -f "$DIR/slot0.bin.dft" ] && du -h "$DIR/slot0.bin.dft" | cut -f1 || echo none)) in $(printf '%s' "$out" | python3 -c "import json,sys
try: print(round(json.load(sys.stdin)['timings']['save_ms']/1000,1))
except Exception: print('?')")s"
}

case "${1:-}" in
autosave)
    [ -e "$DIR/hold" ] && exit 0
    inv=server
    if [ -n "$UNIT" ]; then
        systemctl is-active --quiet "$UNIT" || exit 0
        inv=$(systemctl show -p InvocationID --value "$UNIT")
    fi
    cur=$(slots_json | python3 -c "import json,sys
try:
    s = json.load(sys.stdin)[0]
    print('busy' if s['is_processing'] else '%s %s' % (s['id_task'], s['n_prompt_tokens']))
except Exception: print('')")
    [ -n "$cur" ] && [ "$cur" != busy ] || exit 0
    set -- $cur
    task="$inv-$1"; ntok=$2
    # a task seen for the first time: the agent was active during this period; save on a later tick
    if [ "$task" != "$(cat "$DIR/autosave.seen" 2>/dev/null)" ]; then
        echo "$task" > "$DIR/autosave.seen"
        exit 0
    fi
    [ "$task" = "$(cat "$DIR/autosave.saved" 2>/dev/null)" ] && exit 0
    prev=$(sed -n 's/^n_tokens=//p' "$DIR/slot0.meta" 2>/dev/null)
    if [ -n "$prev" ] && [ "$ntok" -lt $((prev / 2)) ] &&
       [ -n "$(find "$DIR/slot0.meta" -mmin -720 2>/dev/null)" ]; then
        log "slot holds ~$ntok tokens, under half of the last save ($prev); keeping that save"
        echo "$task" > "$DIR/autosave.saved"
        exit 0
    fi
    do_save && echo "$task" > "$DIR/autosave.saved"
    ;;
save)
    if [ -e "$DIR/hold" ]; then
        log "hold set ($(cat "$DIR/hold" 2>/dev/null)); not saving"
        exit 0
    fi
    do_save
    ;;
hold)
    if [ -e "$DIR/hold" ]; then
        log "hold already set ($(cat "$DIR/hold" 2>/dev/null))"
        exit 0
    fi
    do_save
    date -u +%FT%TZ > "$DIR/hold"
    log "hold set; automatic saves are off until 'slot-cache.sh release'"
    ;;
release)
    rm -f "$DIR/hold"
    log "hold released"
    ;;
restore)
    [ -f "$DIR/slot0.bin" ] || { log "nothing saved"; exit 0; }
    if [ ! -r "$SIG_RUN" ] || [ ! -f "$DIR/slot0.sig" ] || ! cmp -s "$SIG_RUN" "$DIR/slot0.sig"; then
        log "saved slot is from a different setup; not restoring"
        exit 0
    fi
    out=$(slot_action restore slot0.bin 300) || { log "restore request failed"; exit 0; }
    n=$(printf '%s' "$out" | json_field n_restored)
    if [ -n "$n" ]; then
        log "restored $n tokens"
    else
        log "restore failed: $(printf '%s' "$out" | head -c 300)"
    fi
    ;;
status)
    ls -la "$DIR" 2>/dev/null
    [ -f "$DIR/slot0.meta" ] && cat "$DIR/slot0.meta"
    if [ -f "$DIR/slot0.sig" ] && [ -r "$SIG_RUN" ]; then
        cmp -s "$SIG_RUN" "$DIR/slot0.sig" && echo "signature: matches the running server" || echo "signature: differs from the running server"
    fi
    [ -e "$DIR/hold" ] && echo "hold: set since $(cat "$DIR/hold")" || echo "hold: none"
    ;;
*)
    echo "usage: slot-cache.sh save|autosave|restore|hold|release|status" >&2
    ;;
esac
exit 0
