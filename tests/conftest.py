"""Shared fixtures for the climepi tests."""

import queue
from collections.abc import Callable, Iterator
from unittest.mock import patch

import pytest


class ServerDocument:
    """Stand-in for a Bokeh server document whose callbacks run on request."""

    session_context = object()

    def __init__(self) -> None:
        self._callbacks: queue.Queue[Callable[[], None]] = queue.Queue()

    def add_next_tick_callback(self, callback: Callable[[], None]) -> None:
        """Queue a callback, as the server would for its next tick."""
        self._callbacks.put(callback)

    def next_callback(self) -> Callable[[], None]:
        """Wait for background work to queue a callback, and return it."""
        return self._callbacks.get(timeout=10)

    def tick(self) -> None:
        """Wait for the next queued callback and run it."""
        self.next_callback()()


@pytest.fixture
def server_document() -> Iterator[ServerDocument]:
    """Run app operations in the background, as they would be in a server session."""
    document = ServerDocument()
    with patch("climepi.app._background.pn") as mock_panel:
        mock_panel.state.curdoc = document
        yield document
