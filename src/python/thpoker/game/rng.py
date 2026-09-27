"""Deterministic random stream shared by the engine, bots, and sessions.

The stream is SHAKE-256 over a key and a block counter, so the same seed gives the same
numbers on every Python version and platform. That is what makes a hand replayable from its
seed alone, which numpy and `random` do not guarantee across versions.
"""

import hashlib
import secrets

_BLOCK_BYTES = 512


class Rng:
    """A seeded, cryptographically strong random stream.

    Raises `ValueError` for a seed that is not a non-negative int.
    """

    __slots__ = ("_key", "_counter", "_buffer", "_position")

    def __init__(self, seed: int):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        self._reset(hashlib.shake_256(b"poker.rng.v1:" + str(seed).encode()).digest(32))

    def _reset(self, key: bytes):
        self._key = key
        self._counter = 0
        self._buffer = b""
        self._position = 0

    def derive(self, *labels: int | str) -> "Rng":
        """Return an independent stream for a labelled purpose (a hand, a seat, a decision).

        Derived streams depend only on this stream's seed and the labels, not on how much of
        this stream has been consumed.
        """
        material = self._key + b"".join(f"\x00{type(x).__name__}:{x}".encode() for x in labels)
        child = Rng.__new__(Rng)
        child._reset(hashlib.shake_256(material).digest(32))
        return child

    def _next_u64(self) -> int:
        if self._position + 8 > len(self._buffer):
            block = self._key + self._counter.to_bytes(8, "big")
            self._buffer = hashlib.shake_256(block).digest(_BLOCK_BYTES)
            self._counter += 1
            self._position = 0
        value = int.from_bytes(self._buffer[self._position : self._position + 8], "big")
        self._position += 8
        return value

    def random(self) -> float:
        """Uniform float in [0, 1) with 53 bits of precision."""
        return (self._next_u64() >> 11) / 9007199254740992.0

    def randbelow(self, n: int) -> int:
        """Uniform int in [0, n), without modulo bias. Raises `ValueError` if n < 1."""
        if n < 1:
            raise ValueError(f"randbelow needs n >= 1, got {n}")
        shift = 64 - n.bit_length()
        while True:
            value = self._next_u64() >> shift
            if value < n:
                return value

    def permutation(self, n: int) -> list[int]:
        """A uniformly random ordering of range(n) (Fisher-Yates)."""
        items = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.randbelow(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


def new_seed() -> int:
    """A fresh unpredictable seed for a new session."""
    return secrets.randbits(63)
