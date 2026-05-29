"""Module for geocoding addresses using the Nominatim geocoder."""

import threading
from functools import lru_cache
from typing import Any

# Lazy imports (PEP 810) on Python 3.15+; no-op on earlier versions.
__lazy_modules__ = ["geopy", "geopy.extra.rate_limiter", "geopy.geocoders"]

from geopy import Location
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

_geocode: None | RateLimiter = None
_geocode_lock = threading.Lock()


def _initialize_geocode() -> None:
    global _geocode
    with _geocode_lock:
        if _geocode is None:
            _geocode = RateLimiter(
                Nominatim(user_agent="climepi", timeout=10).geocode,
                min_delay_seconds=1,
                max_retries=5,
            )


@lru_cache(maxsize=1000, typed=True)
def geocode(query: str, **kwargs: Any) -> Location:
    """
    Geocode an address using the Nominatim geocoder.

    Uses OpenStreetMap data (https://openstreetmap.org/copyright). Always returns
    a single :class:`geopy.Location`; passing ``exactly_one=False`` is rejected,
    and unresolved queries raise :class:`ValueError`.

    Parameters
    ----------
    query : str
        Query string passed to the Nominatim.geocode method (see
        https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.geocode).
    **kwargs:
        Additional keyword arguments passed to the Nominatim.geocode method.

    Returns
    -------
    geopy.Location:
        Return value of the Nominatim.geocode method (see the link above).

    Raises
    ------
    ValueError
        If ``exactly_one=False`` is passed, or if the query cannot be resolved.
    """
    if kwargs.get("exactly_one", True) is False:
        raise ValueError(
            "geocode() always returns a single Location; "
            "'exactly_one=False' is not supported."
        )
    _initialize_geocode()
    assert _geocode is not None, "Geocode service is not initialized."
    result = _geocode(query, **kwargs)
    if result is None:
        raise ValueError(f"Could not geocode query {query!r}.")
    return result
