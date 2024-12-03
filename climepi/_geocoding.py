"""Module for geocoding addresses using the Nominatim geocoder."""

import threading
from functools import lru_cache

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

_geocode = None
_geocode_lock = threading.Lock()


def _initialize_geocode():
    global _geocode
    with _geocode_lock:
        if _geocode is None:
            _geocode = RateLimiter(
                Nominatim(user_agent="climepi", timeout=10).geocode,
                min_delay_seconds=1,
                max_retries=5,
            )


@lru_cache(maxsize=1000)
def geocode(*args, **kwargs):
    """
    Geocode an address using the Nominatim geocoder.

    Uses OpenStreetMap data (https://openstreetmap.org/copyright).

    Parameters
    ----------
    *args, **kwargs:
        Arguments and keyword arguments passed to the Nominatim.geocode method (see
        https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.geocode).

    Returns
    -------
    location:
        Return value of the Nominatim.geocode method (see the link above).
    """
    _initialize_geocode()
    return _geocode(*args, **kwargs)
