#!/usr/bin/env bash
set -euo pipefail
# Apply before any AI container starts. Host ADC traffic does not traverse FORWARD.
iptables -N DOCKER-USER 2>/dev/null || iptables -S DOCKER-USER >/dev/null
iptables -C FORWARD -j DOCKER-USER 2>/dev/null || iptables -I FORWARD 1 -j DOCKER-USER
iptables -C DOCKER-USER -d 169.254.169.254/32 -j REJECT 2>/dev/null || iptables -I DOCKER-USER 1 -d 169.254.169.254/32 -j REJECT
ip6tables -N CAPSTONE_AI_METADATA 2>/dev/null || ip6tables -S CAPSTONE_AI_METADATA >/dev/null
ip6tables -C FORWARD -j CAPSTONE_AI_METADATA 2>/dev/null || ip6tables -I FORWARD 1 -j CAPSTONE_AI_METADATA
ip6tables -C CAPSTONE_AI_METADATA -d fd20:ce::254/128 -j REJECT 2>/dev/null || ip6tables -I CAPSTONE_AI_METADATA 1 -d fd20:ce::254/128 -j REJECT
