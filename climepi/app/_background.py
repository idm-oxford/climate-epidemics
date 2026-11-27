"""Run blocking work outside the Panel document's event loop."""

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, TypeVar

import panel as pn
import param

_T = TypeVar("_T")
_EXECUTOR = ThreadPoolExecutor(thread_name_prefix="climepi")
_LOGGER = logging.getLogger(__name__)


class SessionTask(param.Parameterized):
    """Serialise the operations sharing a session's datasets and plot controls."""

    busy = param.Boolean(default=False)

    def __init__(self) -> None:
        super().__init__()
        self._lock = RLock()
        self._closing = False
        self._future: Future[Any] | None = None
        self._finish_pending: Callable[[], None] | None = None
        self._cleanup: Callable[[], None] | None = None

    def run(
        self,
        work: Callable[[], _T],
        apply: Callable[[_T], None],
        fail: Callable[[Exception], None],
        *,
        discard: Callable[[_T], None] | None = None,
    ) -> None:
        """Deliver a result on the document thread, or run inline without a server."""
        with self._lock:
            if self.busy or self._closing:
                return
            self.busy = True
            doc = pn.state.curdoc
            if doc is None or doc.session_context is None:
                try:
                    apply(work())
                except Exception as exc:
                    fail(exc)
                    raise
                finally:
                    self.busy = False
                return

            def finish(future: Future[_T]) -> None:
                with self._lock:
                    if self._future is not future:
                        return
                    self._future = None
                    self._finish_pending = None
                    if self._closing:
                        cleanup, self._cleanup = self._cleanup, None
                        try:
                            if discard is not None and not future.cancelled():
                                if future.exception() is None:
                                    discard(future.result())
                        finally:
                            if cleanup is not None:
                                cleanup()
                        return
                    try:
                        apply(future.result())
                    except Exception as exc:
                        _LOGGER.exception("App operation failed")
                        fail(exc)
                    finally:
                        self.busy = False

            def completed(future: Future[_T]) -> None:
                with self._lock:
                    if self._closing:
                        finish(future)
                    else:
                        doc.add_next_tick_callback(lambda: finish(future))

            self._future = _EXECUTOR.submit(work)
            future = self._future
            self._finish_pending = lambda: finish(future)
            self._future.add_done_callback(completed)

    def close(self, cleanup: Callable[[], None]) -> None:
        """Release session resources after any worker has finished using them."""
        with self._lock:
            if self._closing:
                return
            self._closing = True
            if self._future is None:
                cleanup()
            else:
                self._cleanup = cleanup
                if self._future.cancel():
                    return
                if self._finish_pending is not None and self._future.done():
                    self._finish_pending()
