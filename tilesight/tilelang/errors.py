"""Refusals: the analysis stops with a code and the missing fact instead of guessing."""

from __future__ import annotations


class Refusal(Exception):
    """A fact the model needs is missing or a structure is outside the supported contract.

    ``code`` is stable and machine-readable; ``message`` names what is missing.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message

    def payload(self) -> dict:
        return {"code": self.code, "message": self.message}
