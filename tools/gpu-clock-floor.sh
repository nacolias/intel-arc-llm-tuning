#!/bin/bash
# SPDX-License-Identifier: MIT
# Pin or restore the compute-engine (GT0) minimum clock of every Intel GPU driven by xe.
#
#   pin      min_freq = rp0 (the hardware maximum; 2800 MHz on an Arc Pro B70) while a model is served.
#            llama.cpp's layer split (-sm layer) leaves each card idle most of the time, so the cards
#            downclock between bursts. On a quad-B70 host pinning raised decode speed on 2026-09-24,
#            and the GTs still enter C6 at true idle, so it costs about 0 W then. Numbers and
#            method: findings/gpu-clock-floor-speeds-layer-split.md.
#   restore  min_freq = rpn (the hardware minimum; 400 MHz on a B70), the driver default.
#
# GPUs are discovered generically: Intel (vendor 0x8086) PCI display-class devices (class 0x0300 or
# 0x0380) that expose the xe sysfs node tile0/gt0/freq0. Only tile 0 / GT 0 is set.
# Run as root, for example from the llama-server systemd unit:
#   ExecStartPre=+/usr/local/sbin/gpu-clock-floor.sh pin
#   ExecStopPost=+/usr/local/sbin/gpu-clock-floor.sh restore
# Never fails the caller: a card that cannot be set is reported and skipped.
set -u
mode=${1:-}
case "$mode" in pin|restore) ;; *) echo "usage: gpu-clock-floor.sh pin|restore" >&2; exit 0 ;; esac
found=0
for dev in /sys/bus/pci/devices/*; do
  [ "$(cat "$dev/vendor" 2>/dev/null)" = 0x8086 ] || continue
  case "$(cat "$dev/class" 2>/dev/null)" in 0x0300*|0x0380*) ;; *) continue ;; esac
  bdf=${dev##*/}
  found=$((found + 1))
  f=$dev/tile0/gt0/freq0
  [ -d "$f" ] || { echo "gpu-clock-floor: $bdf has no $f, skipped" >&2; continue; }
  if [ "$mode" = pin ]; then v=$(cat "$f/rp0_freq"); else v=$(cat "$f/rpn_freq"); fi
  echo "$v" >"$f/min_freq" 2>/dev/null || echo "gpu-clock-floor: could not set $bdf min_freq=$v" >&2
done
[ "$found" -gt 0 ] || echo "gpu-clock-floor: no Intel display-class PCI devices found" >&2
exit 0
