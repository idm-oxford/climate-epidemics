"""Unit tests for the cluster module of the app subpackage."""

from unittest.mock import patch

import pytest

from climepi.app import cluster


@patch("climepi.app.cluster.LocalCluster", autospec=True)
@patch("climepi.app.cluster.time.sleep", side_effect=StopIteration)
def test_start_cluster(mock_sleep, mock_local_cluster, capsys):
    """Unit test for the _start_cluster method."""
    mock_local_cluster.return_value.dashboard_link = "some link"
    with pytest.raises(StopIteration):
        cluster._start_cluster()
    mock_local_cluster.assert_called_once_with(scheduler_port=64719)
    assert (
        "Dask local cluster started. Check the Dask dashboard at some link"
        in capsys.readouterr().out
    )
    mock_sleep.assert_called_once_with(3600)
