from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
import math

Number = Union[int, float]

def _vec(dim: int, value: Optional[Sequence[Number]], name: str) -> Tuple[float, ...]:
    """None이면 0벡터, 아니면 dim길이 튜플로 변환."""
    if value is None:
        return tuple(0.0 for _ in range(dim))
    if len(value) != dim:
        raise ValueError(f"{name}의 길이가 dim({dim})과 다릅니다. (len={len(value)})")
    return tuple(float(x) for x in value)


def _check_dim(dim: int):
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim은 1 이상의 정수여야 합니다.")

def _to_tuple(seq: Sequence[Number], dim: int, name: str) -> Tuple[Number, ...]:
    if len(seq) != dim:
        raise ValueError(f"{name} 길이가 dim({dim})과 다릅니다. (len={len(seq)})")
    return tuple(seq)

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
