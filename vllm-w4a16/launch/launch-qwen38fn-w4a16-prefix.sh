#!/usr/bin/env bash
# The tuned lane (launch-qwen38fn-w4a16-tuned.sh, unchanged) with prefix caching on, in mamba align mode.
# For agent clients that resend a growing conversation: each turn reuses the cached prefix of the last.
# Image: build ../container/Containerfile first (vLLM 8a728663, the commit the tuned lane was validated on).
#
# disable_eagle_block_drop: with MTP on, vLLM 8a728663 cannot tell which KV group is the draft's, flags every
# group (Mamba too) for the trailing-block drop, and a Mamba hit then needs two consecutive checkpoints that align
# mode never keeps; an exact repeat got 0 hits. Keeping the trailing block only affects the draft's acceptance
# (the target verifies every token): count-to-100 stays at ~189 tok/s with 1804/1806 drafts accepted.
SPEC_EXTRA="${SPEC_EXTRA-\"disable_eagle_block_drop\":true}" PREFIX_CACHE=1 exec bash "$(dirname "$0")/launch-qwen38fn-w4a16-tuned.sh"
