"""Version metadata, mirroring `ruptures.version`."""

from ._ruptures_rs import __version__

__all__ = ["__version__", "version", "version_tuple", "__version_tuple__"]

version: str = __version__
__version_tuple__ = version_tuple = tuple(
    int(part) if part.isdigit() else part for part in __version__.split(".")
)
