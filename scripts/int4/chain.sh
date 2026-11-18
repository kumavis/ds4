#!/usr/bin/env bash
# int4 chain: build, model-free tests, queue 1 (gates + headline). Queue 2 (studies) is launched by hand after the scheduler refit.
exec /data/ds4-int4/scripts/int4/build.sh /data/ds4-int4/scripts/int4/tests.sh /data/ds4-int4/scripts/int4/q1.sh
