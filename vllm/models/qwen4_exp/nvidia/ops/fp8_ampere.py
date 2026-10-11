# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software fp8 e4m3fn conversion for GPUs whose Triton backend has no native
e4m3 cast (NVIDIA sm < 89, e.g. Ampere / RTX 3090).

Triton 3.x supports only ``fp8e5`` and ``fp8e4b15`` on sm < 89, so any
``.to(tl.float8e4nv)`` in the QSA kernels fails at compile time. On such GPUs
the fp8 KV cache keeps the e4m3fn *byte format* (identical to Ada/Hopper), but
the Triton kernels receive the cache as ``uint8`` and convert with the integer
helpers below. Both helpers are bit-exact with PyTorch's e4m3fn conversion
(round-to-nearest-even, saturate to +-448, NaN -> 0x7F | sign).

The dtype of the pointer handed to the kernel selects the path at Triton
compile time; no extra constexpr is needed.
"""

import os
from functools import lru_cache

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# fp32 bit patterns used by f32_to_e4m3_bits
_F32_448 = 0x43E00000  # 448.0f  (largest finite e4m3fn)
_F32_2P14 = 0x46800000  # 16384.0f (2^14)
_F32_MIN_NORMAL = 0x3C800000  # 2^-6 (smallest e4m3fn normal)
_F32_REBIAS = 0x3C000000  # (127 - 7) << 23
_F32_INF = 0x7F800000


@lru_cache(maxsize=1)
def emulate_e4m3() -> bool:
    """True when the e4m3 conversions must be emulated in software.

    ``VLLM_QSA_EMULATE_E4M3=1`` / ``=0`` forces the decision (tests);
    otherwise emulate on NVIDIA compute capability < 8.9.
    """
    v = os.environ.get("VLLM_QSA_EMULATE_E4M3")
    if v is not None:
        return v == "1"
    cap = current_platform.get_device_capability()
    return cap is not None and cap.to_int() < 89


@triton.jit
def f32_to_e4m3_bits(x):
    """fp32 (any float input) -> e4m3fn code as uint8.

    Round-to-nearest-even, saturating: |x| >= 448 (incl. inf) -> 0x7E,
    NaN -> 0x7F (sign kept), |x| < 2^-10 -> +-0 (ties to even).
    Bit-exact with ``torch.Tensor.to(torch.float8_e4m3fn)``.
    """
    b = x.to(tl.float32).to(tl.uint32, bitcast=True)
    sign = b & 0x80000000
    a = b ^ sign  # |x| bits
    is_nan = a > 0x7F800000
    a = tl.minimum(a, 0x43E00000)  # clamp |x| to 448.0
    # subnormal result (|x| < 2^-6): adding 2^14 makes fp32's own RNE land on
    # multiples of 2^-9 (= one e4m3 subnormal step); the low mantissa bits of
    # the sum are then the e4m3 code (0..7, or 8 == 2^-6 which is code 0x08).
    sub = (a.to(tl.float32, bitcast=True) + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
    # normal result: rebias exponent 127 -> 7, RNE on the 20 dropped mantissa
    # bits (round bit + sticky via +0x7FFFF, tie -> even via the lsb that stays)
    odd = (a >> 20) & 1
    nrm = (a - 0x3C000000 + 0x7FFFF + odd) >> 20
    r = tl.where(a < 0x3C800000, sub, nrm)
    r = tl.where(is_nan, 0x7F, r)
    return (r | (sign >> 24)).to(tl.uint8)


@triton.jit
def e4m3_bits_to_f16(b):
    """e4m3fn code (uint8) -> fp16 value; exact for every code.

    Normals: fp16 exponent = e + 8 (bias 7 -> 15), mantissa shifted 3 -> 10
    bits. Subnormals (e == 0): m * 2^-9. Codes 0x7F / 0xFF -> NaN.
    """
    c = b.to(tl.uint16)
    s = (c & 0x80) << 8
    e = (c >> 3) & 0xF
    m = c & 0x7
    nrm = (s | ((e + 8) << 10) | (m << 7)).to(tl.float16, bitcast=True)
    # subnormal: m * 2^-9 is exact in fp16; apply the sign through the bits so
    # that -0.0 (code 0x80) keeps its sign (a float negation would fold it).
    sub = (m.to(tl.float32) * 0.001953125).to(tl.float16)
    sub = (sub.to(tl.uint16, bitcast=True) | s).to(tl.float16, bitcast=True)
    nan = (s | 0x7E00).to(tl.float16, bitcast=True)  # quiet NaN, sign kept
    r = tl.where(e == 0, sub, nrm)
    return tl.where((e == 15) & (m == 7), nan, r)
