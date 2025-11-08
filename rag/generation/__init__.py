"""
Generation components for RAG.

Implements:
- BART generator (p_θ component in RAG paper)
- Custom decoding strategies (Thorough/Fast for RAG-Sequence)
"""

from .bart import BARTGenerator
from .decoding import (
    ThoroughDecoding,
    FastDecoding,
    beam_search,
)

__all__ = [
    "BARTGenerator",
    "ThoroughDecoding",
    "FastDecoding",
    "beam_search",
]
