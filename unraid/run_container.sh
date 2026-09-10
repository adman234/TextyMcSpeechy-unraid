#!/bin/bash
# run_container.sh -- all-in-one image variant.
#
# Upstream's version launches the textymcspeechy-piper container via docker
# compose. In this image the dojo and the GPU stack are the same container, so
# there is nothing to launch: run_training.sh calls this, and all it needs to do
# is apply any automatic espeak-ng pronunciation rules.
#
# Custom rules are compiled into the container filesystem, so they are lost when
# the container is recreated (an Unraid "Apply", or an image update). Reapplying
# them on every training run is what makes them stick.
AUTOMATIC_ESPEAK_RULE_SCRIPT="tts_dojo/ESPEAK_RULES/automated_espeak_rules.sh"

if [ -x "$AUTOMATIC_ESPEAK_RULE_SCRIPT" ] || [ -f "$AUTOMATIC_ESPEAK_RULE_SCRIPT" ]; then
    bash "$AUTOMATIC_ESPEAK_RULE_SCRIPT"
fi
