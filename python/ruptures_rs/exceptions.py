"""Exceptions, mirroring `ruptures.exceptions`."""


class NotEnoughPoints(Exception):
    """Raised when a segment is shorter than the cost function's `min_size`."""


class BadSegmentationParameters(Exception):
    """Raised when no segmentation can satisfy the requested parameters."""
