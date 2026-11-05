#!/bin/bash

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The CTest label/name exclusions that upstream applies when running the
# riscv64 tests under QEMU. Sourced by run_suite.sh and list_e2e_tests.sh.
#
# KEEP IN SYNC with build_tools/cmake/test_riscv.sh (upstream; it keeps these
# lists inline, so they cannot be shared with it).
#
# Note: this file must be *sourced* not executed.

RISCV_LABEL_EXCLUDE='^(nodocker|driver=vulkan|driver=metal|driver=cuda|driver=hip|vulkan_uses_vk_khr_shader_float16_int8|requires-filesystem|requires-dtz|noriscv)$'
RISCV_NAME_EXCLUDE='regression_llvm-cpu_lowering_config'
