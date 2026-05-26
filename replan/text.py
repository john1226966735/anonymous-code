import hashlib
import re
from typing import Iterable, List

import torch


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def hash_embed(text: str, dim: int = 128, n_hashes: int = 4) -> torch.Tensor:
    """Deterministic bag-of-words hashing embedding.

    This keeps the submission package self-contained and avoids downloading
    external embedding models for the reviewer sample run.
    """
    vec = torch.zeros(dim, dtype=torch.float32)
    tokens = tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        h = _stable_hash(token)
        for i in range(n_hashes):
            idx = (h >> (i * 8)) % dim
            sign = 1.0 if ((h >> (i * 8 + 4)) & 1) == 0 else -1.0
            vec[idx] += sign
    norm = vec.norm(p=2)
    if norm > 0:
        vec = vec / norm
    return vec


def batch_hash_embed(texts: Iterable[str], dim: int = 128) -> torch.Tensor:
    return torch.stack([hash_embed(text, dim=dim) for text in texts], dim=0)

