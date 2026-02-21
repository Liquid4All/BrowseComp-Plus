"""
Searchers package for different search implementations.
"""

from enum import Enum

from .base import BaseSearcher
from .custom_searcher import CustomSearcher
from .faiss_searcher import FaissSearcher, ReasonIrSearcher


def _load_bm25():
    """Lazy import so pyserini/Java are only required when BM25 is actually used."""
    from .bm25_searcher import BM25Searcher  # noqa: PLC0415
    return BM25Searcher


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    BM25 = ("bm25", "bm25")
    FAISS = ("faiss", FaissSearcher)
    REASONIR = ("reasonir", ReasonIrSearcher)
    CUSTOM = ("custom", CustomSearcher)

    def __init__(self, cli_name, searcher_class):
        self.cli_name = cli_name
        self._searcher_class = searcher_class

    @property
    def searcher_class(self):
        if self._searcher_class == "bm25":
            return _load_bm25()
        return self._searcher_class

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
