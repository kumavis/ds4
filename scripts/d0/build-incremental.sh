#!/usr/bin/env bash
# Incremental strix-halo build: same variables as `make strix-halo`, without -B,
# so only changed objects rebuild. Run inside nix-shell under the GPU lock:
#   flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run scripts/d0/build-incremental.sh
set -eu
cd "$(dirname "$0")/../.."
make ds4 ds4-bench "$@" \
  CORE_OBJS="ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o ds4_engram.o \$(ROCM_MMQ_OBJS) \$(ROCM_HALO_OBJS)" \
  CFLAGS="$(make -s --no-print-directory -f Makefile -f - print-cflags <<<'print-cflags: ; @echo $(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD')" \
  DS4_LINK="$(make -s --no-print-directory -f Makefile -f - print-link <<<'print-link: ; @echo $(HIPCC) $(ROCM_CFLAGS)')" \
  DS4_LINK_LIBS="$(make -s --no-print-directory -f Makefile -f - print-libs <<<'print-libs: ; @echo $(ROCM_LDLIBS)')"
