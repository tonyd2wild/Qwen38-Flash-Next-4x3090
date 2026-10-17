#!/usr/bin/env bash
# The tuned lane (launch-qwen38fn-w4a16-tuned.sh, unchanged) with prefix caching on, in mamba align mode.
# For agent clients that resend a growing conversation: each turn reuses the cached prefix of the last.
# Image: build ../container/Containerfile first (vLLM 8a728663, the commit the tuned lane was validated on).
PREFIX_CACHE=1 exec bash "$(dirname "$0")/launch-qwen38fn-w4a16-tuned.sh"
