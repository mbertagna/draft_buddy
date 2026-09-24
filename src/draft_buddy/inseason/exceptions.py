"""Exceptions raised by in-season snapshot and trade workflows."""


class InSeasonError(Exception):
    """Base error for in-season trade intelligence."""


class SnapshotNotFoundError(InSeasonError):
    """Raised when a league snapshot has not been loaded."""


class NameResolutionError(InSeasonError):
    """Raised when pasted team or player names cannot be matched uniquely.

    Parameters
    ----------
    unmatched : list of str
        Names with no catalog or mapping match.
    ambiguous : list of str
        Names that matched more than one team or player.
    """

    def __init__(self, unmatched: list[str], ambiguous: list[str]) -> None:
        self.unmatched = list(unmatched)
        self.ambiguous = list(ambiguous)
        parts = []
        if self.unmatched:
            parts.append("unmatched: " + ", ".join(self.unmatched))
        if self.ambiguous:
            parts.append("ambiguous: " + ", ".join(self.ambiguous))
        super().__init__("Could not resolve names (" + "; ".join(parts) + ")")


class TradeValidationError(InSeasonError):
    """Raised when a proposed trade references invalid teams or players."""
