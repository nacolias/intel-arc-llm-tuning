#!/bin/bash
# SPDX-License-Identifier: MIT
#
# bmg-aspm-l1.sh: PCIe ASPM L1 on the Intel Arc Pro B70 (Battlemage, BMG-G31) links only.
#
#   apply   L0s and clock PM off on every link. L1 off on every link except those whose downstream
#           device belongs to a Battlemage card: the card's internal upstream port (8086:e2ff, the
#           physical slot link) and the GPU and audio functions under its 8086:e2f0/e2f1 bridges.
#           Then policy powersave. Checks the result and reverts on any mismatch.
#   revert  policy default: every link goes back to the firmware's ASPM setting.
#   status  policy, links with L1 on, and board power per GPU over 10 s.
#
# Measured effect (host quad-b70-5800x-pex88096: 4x Arc Pro B70 behind a Broadcom PEX88096 switch,
# Ryzen 7 5800X, kernel 7.0.0-34-generic, 2026-09-25), model loaded, no requests:
#   board power 44-49 W -> 4-6 W per card, 11 W for the card that drives a lit console;
#   llama.cpp decode and prefill speed unchanged; no new PCIe error-status bits.
# Details and caveats: findings/bmg-aspm-l1-idle-power.md in the intel-arc-llm-tuning repository.
#
# Why it works this way:
# - Under the "default" policy the kernel re-applies the firmware setting, so writing 1 to
#   link/l1_aspm alone changes nothing. The per-link files set a disable mask; the policy write
#   applies it to every link.
# - The kernel parameter pcie_aspm.policy=powersave would also enable ASPM on the switch uplink and
#   on every other capable link. This script keeps L1 on the card links only.
# - Bus numbers are not hardcoded. They change when cards or other devices are added or moved.
# - The device IDs are those of the Arc Pro B70. Check other Battlemage cards with `lspci -nn`
#   before using it on them.
#
# Limits:
# - revert restores the policy only. The disable bits written by apply stay set until reboot, so a
#   link on which the firmware had enabled ASPM stays off (drivers/pci/pcie/aspm.c, v7.0). On the
#   host above the firmware enables ASPM nowhere, so revert returns it to its boot state.
# - status reads the xe hwmon energy counter. That may wake a runtime-suspended card.
# - Run as root. Try apply, status and a benchmark by hand before enabling bmg-aspm-l1.service.
set -u

POL=/sys/module/pcie_aspm/parameters/policy
DEV=/sys/bus/pci/devices

log() { echo "bmg-aspm-l1: $*"; }

# true if device dir $1 is an Intel display-class function (a GPU)
is_intel_gpu() {
    [ "$(cat "$1/vendor")" = 0x8086 ] || return 1
    case "$(cat "$1/class")" in 0x03*) return 0 ;; esac
    return 1
}

# true if the link above device dir $1 belongs to a Battlemage card
is_card_link() {
    local d=$1 p
    [ "$(cat "$d/vendor")" = 0x8086 ] || return 1
    [ "$(cat "$d/device")" = 0xe2ff ] && return 0
    p=$(dirname "$(readlink -f "$d")")
    [ "$(cat "$p/vendor" 2>/dev/null)" = 0x8086 ] || return 1
    case "$(cat "$p/device" 2>/dev/null)" in 0xe2f0|0xe2f1) return 0 ;; esac
    return 1
}

need_policy() {
    [ -w "$POL" ] || { log "$POL is not writable (not root, or ASPM is disabled by the kernel or firmware)"; exit 1; }
}

revert() {
    need_policy
    echo default > "$POL"
    log "policy $(tr -d '\n' < "$POL")"
}

check() {
    local bad=0 on=0 d f v
    for d in "$DEV"/*; do
        [ -e "$d/link/l1_aspm" ] || continue
        v=$(cat "$d/link/l1_aspm")
        if is_card_link "$d"; then
            if [ "$v" = 1 ]; then on=$((on + 1)); else log "L1 is off on card link $(basename "$d")"; bad=1; fi
        elif [ "$v" != 0 ]; then
            log "L1 is on for non-card link $(basename "$d")"; bad=1
        fi
    done
    for f in "$DEV"/*/link/l0s_aspm "$DEV"/*/link/clkpm; do
        [ -e "$f" ] || continue
        [ "$(cat "$f")" = 0 ] || { log "$f is on"; bad=1; }
    done
    grep -q '\[powersave\]' "$POL" || { log "policy is not powersave"; bad=1; }
    [ "$on" -gt 0 ] || { log "no card links found"; bad=1; }
    CARD_LINKS=$on
    return $bad
}

apply() {
    local i d f n
    need_policy
    # xe binds the GPUs after early boot; wait up to 2 minutes for every Intel GPU to have a driver
    for i in $(seq 120); do
        n=0
        for d in "$DEV"/*; do
            is_intel_gpu "$d" && [ ! -e "$d/driver" ] && n=$((n + 1))
        done
        [ "$n" -eq 0 ] && break
        sleep 1
    done
    [ "$n" -eq 0 ] || log "$n Intel GPU(s) still without a driver; applying anyway"

    # the per-link switches only set a disable mask; the policy write below applies them
    for f in "$DEV"/*/link/l0s_aspm "$DEV"/*/link/clkpm; do
        [ -e "$f" ] && echo 0 > "$f"
    done
    for d in "$DEV"/*; do
        [ -e "$d/link/l1_aspm" ] || continue
        if is_card_link "$d"; then echo 1 > "$d/link/l1_aspm"; else echo 0 > "$d/link/l1_aspm"; fi
    done
    echo powersave > "$POL"

    if check; then
        log "L1 on for $CARD_LINKS card links; policy $(tr -d '\n' < "$POL")"
    else
        log "check failed; reverting"
        revert
        return 1
    fi
}

status() {
    local d h a b
    log "policy $(tr -d '\n' < "$POL")"
    for d in "$DEV"/*; do
        [ -e "$d/link/l1_aspm" ] && [ "$(cat "$d/link/l1_aspm")" = 1 ] && echo "  L1 on: $(basename "$d") $(cat "$d/vendor"):$(cat "$d/device")"
    done
    for d in "$DEV"/*; do
        is_intel_gpu "$d" || continue
        h=$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -1)
        [ -n "$h" ] && [ -e "$h/energy1_input" ] || continue
        # energy1_input is board ("card") energy in microjoules
        a=$(cat "$h/energy1_input"); sleep 10; b=$(cat "$h/energy1_input")
        echo "  $(basename "$d"): $(( (b - a) / 10000000 )) W board power (10 s)"
    done
}

case "${1:-}" in
    apply)  apply ;;
    revert) revert ;;
    status) status ;;
    *) echo "usage: bmg-aspm-l1.sh apply|revert|status" >&2; exit 2 ;;
esac
