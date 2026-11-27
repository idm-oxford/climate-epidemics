"""Unit tests for the _background module of the app subpackage."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from climepi.app import _background
from climepi.app._background import SessionTask


def _unexpected(value):
    raise AssertionError(f"Unexpected callback with {value!r}")


def test_run_inline():
    """Test that work runs inline without a server, re-raising errors."""
    task = SessionTask()
    results = []
    task.run(lambda: 1, results.append, _unexpected)
    error = ValueError("no data")

    def broken():
        raise error

    with pytest.raises(ValueError, match="no data"):
        task.run(broken, _unexpected, results.append)
    assert results == [1, error]
    assert not task.busy


def test_run_in_server(server_document):
    """Test that results are applied on a document tick and extra work is ignored."""
    task = SessionTask()
    applied = []
    task.run(lambda: "result", applied.append, _unexpected)
    assert task.busy
    task.run(lambda: "ignored", applied.append, _unexpected)
    server_document.tick()
    assert applied == ["result"]
    assert not task.busy


def test_run_reports_failures(server_document, caplog):
    """Test that errors from work, or from applying its result, are reported."""
    task = SessionTask()
    failures = []

    def broken():
        raise OSError("no data")

    def apply(_result):
        raise ValueError("cannot show")

    task.run(broken, _unexpected, failures.append)
    server_document.tick()
    task.run(lambda: 1, apply, failures.append)
    server_document.tick()
    assert [str(failure) for failure in failures] == ["no data", "cannot show"]
    assert "App operation failed" in caplog.text
    assert not task.busy


def test_close_waits_for_running_work(server_document):
    """Test that closing during work cleans up once the work has finished."""
    task = SessionTask()
    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    discarded = []

    def work():
        started.set()
        assert release.wait(10)
        return "result"

    task.run(work, _unexpected, _unexpected, discard=discarded.append)
    assert started.wait(10)
    task.close(cleaned.set)
    task.close(_unexpected)  # closing again has no effect
    assert not cleaned.is_set()
    release.set()
    assert cleaned.wait(10)
    assert discarded == ["result"]


def test_close_discards_result_awaiting_tick(server_document):
    """Test that a finished result is discarded if the session closes before a tick."""
    task = SessionTask()
    discarded, cleaned = [], []
    task.run(lambda: "result", _unexpected, _unexpected, discard=discarded.append)
    late_tick = server_document.next_callback()
    task.close(lambda: cleaned.append(True))
    assert discarded == ["result"]
    assert cleaned == [True]
    late_tick()  # the result has already been handled


def test_close_cancels_queued_work(server_document):
    """Test that closing cancels work that has not started."""
    blocker = threading.Event()
    ran, cleaned = [], []
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(blocker.wait, 10)
        with patch.object(_background, "_EXECUTOR", executor):
            task = SessionTask()
            task.run(lambda: ran.append(True), _unexpected, _unexpected)
            task.close(lambda: cleaned.append(True))
        blocker.set()
    assert cleaned == [True]
    assert not ran


def test_close_idle():
    """Test that closing an idle task cleans up immediately."""
    task = SessionTask()
    cleaned = []
    task.close(lambda: cleaned.append(True))
    assert cleaned == [True]
    task.run(lambda: 1, _unexpected, _unexpected)  # closed tasks run nothing
