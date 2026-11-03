#!/bin/bash
# E2: historical G1 / G2 (shared runner for B2 and E2, see ../../shared/g12_fixed.sh). Usage as there.
exec bash $(dirname $0)/../../shared/g12_fixed.sh "$@"
