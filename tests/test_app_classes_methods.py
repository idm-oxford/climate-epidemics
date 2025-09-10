"""Unit tests for the _app_classes_methods module of the app subpackage."""

import pathlib
import tempfile
from unittest.mock import patch

import dask
import holoviews as hv
import netCDF4  # noqa (avoids warning https://github.com/pydata/xarray/issues/7259)
import numpy as np
import numpy.testing as npt
import panel as pn
import param
import pytest
import xarray as xr
import xarray.testing as xrt
from holoviews.element.comparison import Comparison as hvt

import climepi  # noqa
import climepi.app._app_classes_methods as app_classes_methods
from climepi import epimod
from climepi.testing.fixtures import generate_dataset

dask.config.set(scheduler="synchronous")  # enforce synchronous scheduler


@pytest.fixture(autouse=True)
def cache_cleanup():
    """Clean up the cache after each test."""
    pn.state.clear_caches()


original_plot_map = climepi.ClimEpiDatasetAccessor.plot_map


def _plot_map(self, *args, **kwargs):
    """
    Run dataset.climepi.plot_map method but imposing rasterize=False.

    This is needed because the default rasterize=True option seems to cause an error
    when in debug mode.
    """
    return original_plot_map(self, *args, **{**kwargs, "rasterize": False})


@patch.object(xr, "open_mfdataset", autospec=True)
@patch("climepi.app._app_classes_methods.climdata.get_example_dataset", autospec=True)
def test_load_clim_data_func(mock_get_example_dataset, mock_open_mfdataset):
    """Unit test for the _load_clim_data_func function."""
    mock_get_example_dataset.return_value = "mocked_example_dataset"
    mock_open_mfdataset.return_value = xr.Dataset(
        {"time_bnds": (("time", "bnds"), np.array([[0, 1, 2], [1, 2, 3]]))},
    ).chunk({"time": 3})
    result = app_classes_methods._load_clim_data_func(
        "Example dataset",
        clim_example_name="some_example_name",
        clim_example_base_dir="some/dir",
    )
    mock_get_example_dataset.assert_called_once_with(
        "some_example_name", base_dir="some/dir"
    )
    assert result == "mocked_example_dataset"
    # Check cached version is returned if the same example_name and base_dir are
    # provided
    mock_get_example_dataset.return_value = "another_mocked_dataset"
    result_cached = app_classes_methods._load_clim_data_func(
        "Example dataset",
        clim_example_name="some_example_name",
        clim_example_base_dir="some/dir",
    )
    assert result_cached == "mocked_example_dataset"
    mock_get_example_dataset.assert_called_once()
    # Test with custom dataset option
    result_custom = app_classes_methods._load_clim_data_func(
        "Custom dataset", custom_clim_data_dir="path/to/custom/data"
    )
    assert result_custom == mock_open_mfdataset.return_value
    assert not result_custom.chunks  # method loads time_bnds into memory
    mock_open_mfdataset.assert_called_once_with(
        "path/to/custom/data/*.nc",
        data_vars="minimal",
        chunks={},
        coords="minimal",
        compat="override",
    )
    # Test with invalid dataset option
    with pytest.raises(ValueError, match="Unrecognised climate data option"):
        app_classes_methods._load_clim_data_func("Not an option")


@patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
def test_get_epi_model_func(mock_get_example_model):
    """Unit test for the _get_epi_model_func function."""
    # Test with example_name provided
    mock_get_example_model.return_value = "mocked_model"
    result_named = app_classes_methods._get_epi_model_func(
        example_name="some_example_name"
    )
    mock_get_example_model.assert_called_once_with("some_example_name")
    assert result_named == "mocked_model"
    # Test with temperature_range provided
    result_temp_range = app_classes_methods._get_epi_model_func(
        temperature_range=(15, 30)
    )
    assert isinstance(result_temp_range, epimod.SuitabilityModel)
    assert result_temp_range.temperature_range == (15, 30)
    # Check error if either both or neither of example_name and temperature_range are
    # provided
    with pytest.raises(
        ValueError,
        match="Exactly one of example_name and temperature_range must be provided",
    ):
        app_classes_methods._get_epi_model_func(
            example_name="another_name", temperature_range=(0, 10)
        )
    with pytest.raises(
        ValueError,
        match="Exactly one of example_name and temperature_range must be provided",
    ):
        app_classes_methods._get_epi_model_func()


@patch("climepi.app._app_classes_methods._compute_to_file_reopen", autospec=True)
@patch.object(pathlib.Path, "unlink", autospec=True)
def test_run_epi_model_func(mock_path_unlink, mock_compute_to_file_reopen):
    """Unit test for the _run_epi_model_func function."""

    def _mock_compute_to_file_reopen(ds_in, save_path):
        return ds_in

    mock_compute_to_file_reopen.side_effect = _mock_compute_to_file_reopen

    ds_clim = generate_dataset(data_var="temperature", frequency="monthly")
    ds_clim["temperature"].values = 30 * np.random.rand(*ds_clim["temperature"].shape)
    epi_model = epimod.SuitabilityModel(temperature_range=(15, 30))

    result = app_classes_methods._run_epi_model_func(
        ds_clim,
        epi_model,
        return_yearly_portion_suitable=True,
        save_path=pathlib.Path("some/dir/ds_out.nc"),
    )
    xrt.assert_identical(
        result, epi_model.run(ds_clim, return_yearly_portion_suitable=True)
    )
    assert mock_compute_to_file_reopen.call_count == 2
    mock_path_unlink.assert_called_once_with(pathlib.Path("some/dir/ds_suitability.nc"))


@pytest.mark.parametrize("temporal", ["daily", "monthly", "yearly"])
@pytest.mark.parametrize("spatial", ["single", "list", "grid"])
@pytest.mark.parametrize("ensemble", ["single", "multiple"])
@pytest.mark.parametrize("scenario", ["single", "multiple"])
@pytest.mark.parametrize("model", ["single", "multiple"])
def test_get_scope_dict(temporal, spatial, ensemble, scenario, model):
    """Unit test for the _get_scope_dict function."""
    ds = generate_dataset(
        data_var="temperature",
        frequency=temporal,
        extra_dims={"realization": 3, "scenario": 2, "model": 2},
    )
    if spatial == "single":
        ds = ds.isel(lat=0, lon=0)
    elif spatial == "list":
        ds = ds.climepi.sel_geo(["lords", "gabba"])
    if ensemble == "single":
        ds = ds.isel(realization=0, drop=True)
    if scenario == "single":
        ds = ds.isel(scenario=0)
    if model == "single":
        ds = ds.isel(model=[0])
    result = app_classes_methods._get_scope_dict(ds)
    expected = {
        "temporal": temporal,
        "spatial": spatial,
        "ensemble": ensemble,
        "scenario": scenario,
        "model": model,
    }
    assert result == expected


def test_get_view_func():
    """
    Unit test for the _get_view_func function.

    Since the function is a wrapper around the generate_plot method of the _Plotter
    class, we only check that the function returns the same result as applying the
    generate_plot method to a _Plotter instance.
    """
    ds = generate_dataset(data_var="temperature", frequency="monthly")
    plot_settings = {
        "plot_type": "time series",
        "data_var": "temperature",
        "temporal_scope": "monthly",
        "year_range": [2000, 2001],
        "location_string": "not used",
        "realization": "not used",
        "ensemble_stat": "not used",
        "model": "not used",
        "scenario": "not used",
    }
    result = app_classes_methods._get_view_func(ds, plot_settings)

    # Check that the result is the same as the view attribute of a _Plotter instance
    # with the same inputs after calling the generate_plot method
    plotter = app_classes_methods._Plotter(ds_in=ds, plot_settings=plot_settings)
    plotter.generate_plot()
    expected = plotter.view
    assert isinstance(result[1].object, hv.Overlay)
    hvt.assertEqual(result[1].object, expected[1].object)


def test_compute_to_file_reopen():
    """Unit test for the _compute_to_file_reopen function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_in = generate_dataset(data_var="temperature", frequency="monthly")
        ds_in["temperature"] = ds_in["temperature"].chunk(
            {"time": 2, "lat": 1, "lon": 1}
        )
        save_path = pathlib.Path(tmpdir) / "test.nc"

        ds_out = app_classes_methods._compute_to_file_reopen(ds_in, save_path)
        ds_out["time_bnds"] = ds_out["time_bnds"].compute()

        assert pathlib.Path(save_path).exists(), "File not saved."
        xrt.assert_identical(ds_in, ds_out)
        for var in ["time", "lat", "lon"]:
            assert ds_out.chunks[var] == ds_in.chunks[var], f"Chunks for {var} differ."

        ds_out.close()


class TestPlotter:
    """Unit tests for the _Plotter class."""

    def test_init(self):
        """Unit test for the __init__ method."""
        ds_in = generate_dataset(
            data_var="temperature", frequency="monthly", extra_dims={"realization": 3}
        )
        plot_settings = {"swing": "miss"}
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        assert plotter.view is None
        xrt.assert_identical(plotter._ds_base, ds_in)
        assert plotter._scope_dict_base == {
            "temporal": "monthly",
            "spatial": "grid",
            "ensemble": "multiple",
            "scenario": "single",
            "model": "single",
        }
        assert plotter._plot_settings == plot_settings
        assert plotter._ds_plot is None

    @patch(
        "climepi.app._app_classes_methods.ClimEpiDatasetAccessor.plot_map",
        new=_plot_map,
    )
    @pytest.mark.parametrize(
        "plot_type",
        ["time series", "map", "variance decomposition", "unsupported_type"],
    )
    def test_generate_plot(self, plot_type):
        """Unit test for the generate_plot method."""
        ds_in = generate_dataset(
            data_var="temperature", frequency="monthly", extra_dims={"realization": 3}
        )
        plot_settings = {
            "plot_type": plot_type,
            "data_var": "temperature",
            "temporal_scope": "monthly",
            "year_range": [2000, 2001],
            "location_string": "SCG",
            "realization": "all",
            "model": "not used",
            "scenario": "not used",
            "ensemble_stat": "mean",
        }

        # Mock ds.climepi.plot_map to set rasterize=False (true causes failure )

        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        if plot_type == "unsupported_type":
            # Check error handling for unsupported plot type (patch _get_ds_plot to
            # avoid that method raising an error first)
            with patch.object(plotter, "_get_ds_plot", autospec=True):
                plotter._ds_plot = xr.Dataset()
                with pytest.raises(
                    ValueError, match="Unsupported plot type: unsupported_type"
                ):
                    plotter.generate_plot()
                return
        plotter.generate_plot()
        if plot_type == "map":
            view_panel = plotter.view[1][0]
        else:
            view_panel = plotter.view[1]
        assert isinstance(view_panel, pn.pane.HoloViews)
        assert not view_panel.linked_axes
        assert view_panel.widget_location == "bottom"
        assert view_panel.center
        plot_obj = view_panel.object
        if plot_type == "map":
            assert isinstance(plot_obj, hv.HoloMap)
            assert plot_obj.kdims == [hv.Dimension("time")]
            plot_obj_last = plot_obj.last
            assert isinstance(plot_obj_last, hv.Overlay)
            assert plot_obj_last.keys() == [
                ("QuadMesh", "I"),
                ("Coastline", "I"),
                ("Borders", "I"),
                ("Ocean", "I"),
                ("Lakes", "I"),
            ]
        elif plot_type == "time series":
            assert isinstance(plot_obj, hv.Overlay)
            assert plot_obj.keys() == [
                ("Area", "Internal_variability"),
                ("Curve", "Mean"),
                ("Curve", "Example_trajectory"),
            ]
        elif plot_type == "variance decomposition":
            assert isinstance(plot_obj, hv.Layout)
            assert not plot_obj.opts["shared_axes"]
            assert (
                list(plot_obj.NdOverlay.I.data.keys())
                == list(plot_obj.NdOverlay.II.data.keys())
                == [
                    ("Internal variability",),
                    ("Model uncertainty",),
                    ("Scenario uncertainty",),
                ]
            )
            # Second fractional plot should show all variance being from internal
            # variability
            npt.assert_allclose(
                plot_obj.NdOverlay.II.data[("Internal variability",)].data[
                    "Internal variability"
                ],
                1,
            )
        else:
            raise ValueError("Unexpected plot type provided to test.")

    def test_get_ds_plot(self):
        """Unit test for the _get_ds_plot method."""
        ds_in = generate_dataset(
            data_var=["temperature", "precipitation"],
            frequency="monthly",
            extra_dims={
                "realization": [0, 1, 2],
                "model": ["a", "b"],
                "scenario": ["x", "y"],
            },
        )
        plot_settings = {
            "plot_type": "time series",
            "data_var": "temperature",
            "temporal_scope": "yearly",
            "year_range": [2000, 2000],
            "location_string": "SCG",
            "realization": 1,
            "model": "a",
            "scenario": "y",
            "ensemble_stat": "unused",
        }
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        plotter._get_ds_plot()
        xrt.assert_identical(
            plotter._ds_plot,
            ds_in[["temperature", "time_bnds", "lat_bnds", "lon_bnds"]]
            .climepi.sel_geo("SCG")
            .isel(time=ds_in.time.dt.year == 2000)
            .sel(realization=1, model="a", scenario="y")
            .climepi.yearly_average(),
        )

    def test_sel_data_var_ds_plot(self):
        """Unit test for the _sel_data_var_ds_plot method."""
        ds_in = generate_dataset(data_var=["temperature", "precipitation"])
        plot_settings = {"data_var": "precipitation"}
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        plotter._ds_plot = ds_in
        plotter._sel_data_var_ds_plot()
        xrt.assert_identical(
            plotter._ds_plot,
            ds_in[["precipitation", "time_bnds", "lat_bnds", "lon_bnds"]],
        )

    @pytest.mark.parametrize("plot_type", ["time series", "map", "fake type"])
    @pytest.mark.parametrize("spatial_scope_base", ["single", "list", "grid"])
    def test_spatial_index_ds_plot(self, plot_type, spatial_scope_base):
        """Unit test for the _spatial_index_ds_plot method."""
        ds_in = generate_dataset(data_var="temperature")
        if spatial_scope_base == "single":
            ds_in = ds_in.isel(lat=0, lon=0)
        elif spatial_scope_base == "list":
            ds_in = ds_in.climepi.sel_geo(["Lords", "SCG"])
        plot_settings = {
            "plot_type": plot_type,
            "location_selection": "Lords",  # only used for time series+list
            "location_string": "Gabba",  # only used for time series+grid
        }
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        plotter._ds_plot = ds_in
        if plot_type == "fake type" and spatial_scope_base == "grid":
            with pytest.raises(ValueError, match="Unsupported"):
                plotter._spatial_index_ds_plot()
            return
        plotter._spatial_index_ds_plot()
        if plot_type in ["time series", "fake type"] and spatial_scope_base == "single":
            xrt.assert_identical(plotter._ds_plot, ds_in)
        elif plot_type in ["time series", "fake type"] and spatial_scope_base == "list":
            xrt.assert_identical(plotter._ds_plot, ds_in.sel(location="Lords"))
        elif plot_type == "time series" and spatial_scope_base == "grid":
            xrt.assert_identical(plotter._ds_plot, ds_in.climepi.sel_geo("Gabba"))
        elif plot_type == "map":
            xrt.assert_identical(plotter._ds_plot, ds_in)
        else:
            raise ValueError(
                "Unexpected combination of plot_type and spatial_scope_base provided "
                f"to test: {plot_type}, {spatial_scope_base}"
            )

    @pytest.mark.parametrize("temporal_scope", ["yearly", "difference between years"])
    @pytest.mark.parametrize("year_range", [[2000, 2001], [2000, 2002], [2000, 2003]])
    def test_temporal_index_ds_plot(self, temporal_scope, year_range):
        """Unit test for the _temporal_index_ds_plot method."""
        ds_in = generate_dataset(data_var="temperature", frequency="yearly")
        assert list(ds_in.time.dt.year.values) == [2000, 2001, 2002]
        plot_settings = {"temporal_scope": temporal_scope, "year_range": year_range}
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        plotter._ds_plot = ds_in
        if temporal_scope == "difference between years" and year_range == [
            2000,
            2003,
        ]:
            with pytest.raises(ValueError, match="Only years in the dataset"):
                plotter._temporal_index_ds_plot()
            return
        plotter._temporal_index_ds_plot()
        if year_range == [2000, 2001]:
            xrt.assert_identical(plotter._ds_plot, ds_in.isel(time=[0, 1]))
        elif temporal_scope == "yearly" and year_range in [[2000, 2002], [2000, 2003]]:
            xrt.assert_identical(plotter._ds_plot, ds_in)
        elif temporal_scope == "difference between years" and year_range == [
            2000,
            2002,
        ]:
            xrt.assert_identical(plotter._ds_plot, ds_in.isel(time=[0, 2]))
        else:
            raise ValueError(
                "Unexpected combination of temporal_scope and year_range provided to "
                f"test: {temporal_scope}, {year_range}"
            )

    @pytest.mark.parametrize("dim", ["realization", "model", "scenario"])
    @pytest.mark.parametrize("scope_base", ["single", "multiple"])
    @pytest.mark.parametrize("sel_plot", ["single", "all"])
    def test_ensemble_model_scenario_index_ds_plot(self, dim, scope_base, sel_plot):
        """
        Test _ensemble_index_ds_plot, _model_index_ds_plot, and _scenario_index_ds_plot.

        The three methods are tested together because they are very similar.
        """
        ds_in = generate_dataset(
            data_var="temperature", extra_dims={dim: ["a", "b", "c"]}
        )
        if scope_base == "single":
            ds_in = ds_in.isel(**{dim: 0})
        plot_settings = {
            "model": "a",
            "scenario": "b",
            "realization": "c",
        }
        if sel_plot == "all":
            plot_settings.update({dim: "all"})
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        plotter._ds_plot = ds_in
        if dim == "realization":
            plotter._ensemble_index_ds_plot()
        elif dim == "model":
            plotter._model_index_ds_plot()
        elif dim == "scenario":
            plotter._scenario_index_ds_plot()
        else:
            raise ValueError(f"Unexpected dimension provided to test: {dim}")
        if scope_base == "single" or sel_plot == "all":
            xrt.assert_identical(plotter._ds_plot, ds_in)
        elif dim == "realization" and sel_plot == "single":
            xrt.assert_identical(plotter._ds_plot, ds_in.sel(realization="c"))
        elif dim == "model" and sel_plot == "single":
            xrt.assert_identical(plotter._ds_plot, ds_in.sel(model="a"))
        elif dim == "scenario" and sel_plot == "single":
            xrt.assert_identical(plotter._ds_plot, ds_in.sel(scenario="b"))
        else:
            raise ValueError(
                "Unexpected combination of dimension, scope_base, and sel_plot "
                f"provided to test: {dim}, {scope_base}, {sel_plot}"
            )

    @pytest.mark.parametrize("temporal_scope", ["yearly", "difference between years"])
    @pytest.mark.parametrize("temporal_scope_base", ["monthly", "yearly"])
    def test_temporal_ops_ds_plot(self, temporal_scope, temporal_scope_base):
        """Unit test for the _temporal_ops_ds_plot method."""
        ds_in = generate_dataset(data_var="temperature", frequency=temporal_scope_base)
        if temporal_scope == "difference between years":
            ds_in = ds_in.isel(time=ds_in.time.dt.year.isin([2000, 2001]))
        plot_settings = {"temporal_scope": temporal_scope, "year_range": [2000, 2001]}
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        plotter._ds_plot = ds_in
        plotter._temporal_ops_ds_plot()
        if temporal_scope == "yearly" and temporal_scope_base == "monthly":
            xrt.assert_identical(plotter._ds_plot, ds_in.climepi.yearly_average())
        elif temporal_scope == "yearly" and temporal_scope_base == "yearly":
            xrt.assert_identical(plotter._ds_plot, ds_in)
        elif temporal_scope == "difference between years":
            assert "time" not in plotter._ds_plot
            assert "time_bnds" not in plotter._ds_plot
            ds_yearly = ds_in.climepi.yearly_average()
            npt.assert_equal(
                plotter._ds_plot["temperature"].values,
                ds_yearly.isel(time=1)["temperature"].values
                - ds_yearly.isel(time=0)["temperature"].values,
            )
        else:
            raise ValueError(
                "Unexpected combination of temporal_scope and temporal_scope_base "
                f"provided to test: {temporal_scope}, {temporal_scope_base}"
            )

    @pytest.mark.parametrize("plot_type", ["time series", "map"])
    @pytest.mark.parametrize(
        "ensemble_stat", ["mean", "std", "min", "max", "individual realization(s)"]
    )
    def test_ensemble_ops_ds_plot(self, plot_type, ensemble_stat):
        """Unit test for the _ensemble_ops_ds_plot method."""
        ds_in = generate_dataset(data_var="temperature", extra_dims={"realization": 3})
        plot_settings = {"plot_type": plot_type, "ensemble_stat": ensemble_stat}
        plotter = app_classes_methods._Plotter(ds_in=ds_in, plot_settings=plot_settings)
        plotter._ds_plot = ds_in
        plotter._ensemble_ops_ds_plot()
        if plot_type == "time series" or ensemble_stat == "individual realization(s)":
            xrt.assert_identical(plotter._ds_plot, ds_in)
        elif plot_type == "map":
            xrt.assert_identical(
                plotter._ds_plot, ds_in.climepi.ensemble_stats().sel(stat=ensemble_stat)
            )
        else:
            raise ValueError(
                "Unexpected combination of plot_type and ensemble_stat provided to "
                f"test: {plot_type}, {ensemble_stat}"
            )


class TestPlotController:
    """Unit tests for the _PlotController class."""

    @pytest.mark.parametrize("ds_provided", [True, False])
    def test_init(self, ds_provided):
        """Unit test for the __init__ method."""
        if ds_provided:
            ds_in = generate_dataset(data_var="temperature", frequency="monthly")
            plot_controller = app_classes_methods._PlotController(ds_in=ds_in)
        else:
            plot_controller = app_classes_methods._PlotController()
        for attr, value in [
            ("plot_type", "time series" if ds_provided else None),
            ("data_var", "temperature" if ds_provided else None),
            ("location_string", "[Type location]"),
            ("location_selection", None),
            ("temporal_scope", "yearly" if ds_provided else None),
            ("year_range", (2000, 2001) if ds_provided else None),
            ("scenario", None),
            ("model", None),
            ("realization", None),
            ("ensemble_stat", "mean" if ds_provided else None),
            ("plot_initiator", False),
            ("plot_generated", False),
            ("plot_status", "Plot not yet generated"),
            ("view_refresher", False),
            ("_ds_base", ds_in if ds_provided else None),
            (
                "_scope_dict_base",
                app_classes_methods._get_scope_dict(ds_in) if ds_provided else None,
            ),
        ]:
            assert getattr(plot_controller, attr) == value, (
                f"Unexpected value for {attr}: expected {value}, "
                f"got {getattr(plot_controller, attr)}"
            )
        assert len(plot_controller.view) == 0
        assert len(plot_controller.controls) == (1 if ds_provided else 0)

    @pytest.mark.parametrize("ds_option", ["provide", "replace", "remove"])
    def test_initialize(self, ds_option):
        """Unit test for the initialize method."""
        if ds_option in ["replace", "remove"]:
            ds_old = generate_dataset(data_var="temperature", frequency="monthly")
            plot_controller = app_classes_methods._PlotController(ds_in=ds_old)
            plot_controller.location_string = "SCG"
            plot_controller._update_view()
            assert len(plot_controller.view) == 1
        elif ds_option == "provide":
            plot_controller = app_classes_methods._PlotController()
        else:
            raise ValueError(f"Unexpected ds_option: {ds_option} provided to test.")
        if ds_option in ["provide", "replace"]:
            ds_new = generate_dataset(data_var="precipitation", frequency="daily")
            plot_controller.initialize(ds_new)
            assert plot_controller._ds_base is ds_new
            assert (
                plot_controller._scope_dict_base
                == app_classes_methods._get_scope_dict(ds_new)
            )
            assert plot_controller.data_var == "precipitation"
            assert plot_controller.year_range == (  # daily dataset has only one year
                2000,
                2000,
            )
            assert plot_controller.location_string == "[Type location]"
            assert len(plot_controller.controls) == 1
            assert plot_controller.controls[0].widgets is not None
            assert len(plot_controller.view) == 0
        elif ds_option == "remove":
            plot_controller.initialize()
            assert plot_controller._ds_base is None
            assert plot_controller._scope_dict_base is None
            assert len(plot_controller.controls) == 0
            assert len(plot_controller.view) == 0
        else:
            raise ValueError(f"Unexpected ds_option: {ds_option} provided to test.")

    @pytest.mark.parametrize(
        "temporal_scope_base,spatial_scope_base,scenario_scope_base,"
        "model_scope_base,ensemble_scope_base",
        [
            ("daily", "list", "single", "single", "single"),
            ("monthly", "grid", "multiple", "multiple", "multiple"),
            ("yearly", "list", "single", "single", "multiple"),
            ("fake option", "list", "single", "single", "single"),
            ("daily", "fake option", "single", "single", "single"),
            ("daily", "list", "fake option", "single", "single"),
            ("daily", "list", "single", "fake option", "single"),
            ("daily", "list", "single", "single", "fake option"),
        ],
    )
    def test_initialize_params(
        self,
        temporal_scope_base,
        spatial_scope_base,
        scenario_scope_base,
        model_scope_base,
        ensemble_scope_base,
    ):
        """Unit test for the _initialize_params method."""
        ds_in = generate_dataset(
            data_var=["precipitation", "temperature"],
            frequency=temporal_scope_base
            if temporal_scope_base != "fake option"
            else "daily",
            extra_dims={
                "scenario": 2 if scenario_scope_base == "multiple" else 1,
                "model": 2 if model_scope_base == "multiple" else 1,
                "realization": 3 if ensemble_scope_base == "multiple" else 1,
            },
        )
        if temporal_scope_base == "yearly":
            ds_in = ds_in.isel(time=ds_in.time.dt.year.isin([2000, 2002]))
        if spatial_scope_base == "list":
            ds_in = ds_in.climepi.sel_geo(["Lords", "SCG"])

        scope_dict = app_classes_methods._get_scope_dict(ds_in)

        if temporal_scope_base == "fake option":
            scope_dict["temporal"] = "fake option"
        if spatial_scope_base == "fake option":
            scope_dict["spatial"] = "fake option"
        if scenario_scope_base == "fake option":
            scope_dict["scenario"] = "fake option"
        if model_scope_base == "fake option":
            scope_dict["model"] = "fake option"
        if ensemble_scope_base == "fake option":
            scope_dict["ensemble"] = "fake option"

        plot_controller = app_classes_methods._PlotController()
        plot_controller._ds_base = ds_in
        plot_controller._scope_dict_base = scope_dict

        if "fake option" in [
            temporal_scope_base,
            spatial_scope_base,
            scenario_scope_base,
            model_scope_base,
            ensemble_scope_base,
        ]:
            with pytest.raises(ValueError, match="Unrecognised"):
                plot_controller._initialize_params()
            return

        plot_controller._initialize_params()

        for attr, value in [
            ("plot_type", "time series"),
            ("data_var", "precipitation"),
            ("location_string", "[Type location]"),
            ("location_selection", "all" if spatial_scope_base == "list" else None),
            ("temporal_scope", "yearly"),
            ("scenario", "all" if scenario_scope_base == "multiple" else None),
            ("model", "all" if model_scope_base == "multiple" else None),
            ("realization", "all" if ensemble_scope_base == "multiple" else None),
            ("ensemble_stat", "mean"),
        ]:
            assert getattr(plot_controller, attr) == value, (
                f"Unexpected value for {attr}: expected {value}, "
                f"got {getattr(plot_controller, attr)}"
            )
        if temporal_scope_base == "yearly":
            assert set(ds_in.time.dt.year.values) == {2000, 2002}
            assert plot_controller.year_range == (2000, 2002)
            assert plot_controller.param.year_range.step == 2
            assert plot_controller.param.temporal_scope.objects == ["yearly"]
        elif temporal_scope_base == "monthly":
            assert set(ds_in.time.dt.year.values) == {2000, 2001}
            assert plot_controller.year_range == (2000, 2001)
            assert plot_controller.param.year_range.step == 1
            assert plot_controller.param.temporal_scope.objects == ["yearly", "monthly"]
        elif temporal_scope_base == "daily":
            assert set(ds_in.time.dt.year.values) == {2000}
            assert plot_controller.year_range == (2000, 2000)
            assert plot_controller.param.year_range.step == 1
            assert plot_controller.param.temporal_scope.objects == [
                "yearly",
                "monthly",
                "daily",
            ]
        else:
            raise ValueError(
                f"Unexpected temporal_scope_base: {temporal_scope_base} provided."
            )
        if spatial_scope_base == "list":
            assert plot_controller.param.location_selection.objects == [
                "all",
                "Lords",
                "SCG",
            ]
            assert plot_controller.param.location_selection.precedence == 1
            assert plot_controller.param.location_string.precedence == -1
        elif spatial_scope_base == "grid":
            assert plot_controller.param.location_selection.precedence == -1
            assert plot_controller.param.location_string.precedence == 1
        else:
            raise ValueError(
                f"Unexpected spatial_scope_base: {spatial_scope_base} provided."
            )
        assert plot_controller.param.ensemble_stat.precedence == -1

    def test_update_view(self):
        """
        Unit test for the _update_view method.

        The method is triggered through the "plot_initiator" event parameter.
        """
        ds_in = generate_dataset(
            data_var="temperature", frequency="monthly"
        ).climepi.sel_geo("SCG")
        plot_controller = app_classes_methods._PlotController(ds_in=ds_in)
        plot_controller.view.append("some_view")

        view_refresher_trigger_count = 0

        @param.depends(plot_controller.param.view_refresher, watch=True)
        def _update_view_refresher_triggers(view_refresher):
            nonlocal view_refresher_trigger_count
            view_refresher_trigger_count += 1

        plot_controller.param.trigger("plot_initiator")

        assert len(plot_controller.view) == 1
        hvt.assertEqual(
            plot_controller.view[0][1].object,
            app_classes_methods._get_view_func(
                ds_in=ds_in,
                plot_settings={
                    "plot_type": "time series",
                    "data_var": "temperature",
                    "temporal_scope": "yearly",
                    "year_range": [2000, 2001],
                    "location_selection": "not used",
                    "location_string": "not used",
                    "realization": "not used",
                    "ensemble_stat": "not used",
                    "model": "not used",
                    "scenario": "not used",
                },
            )[1].object,
        )
        assert plot_controller.plot_generated
        assert plot_controller.plot_status == "Plot generated"
        assert view_refresher_trigger_count == 2  # refreshed at start/end of generation

        # Check that the view is not updated if the plot has already been generated
        plot_controller.param.trigger("plot_initiator")
        assert view_refresher_trigger_count == 2

        # Check error handling
        plot_controller.initialize(ds_in)
        with patch(
            "climepi.app._app_classes_methods._get_view_func",
            side_effect=ValueError("Some error"),
        ):
            with pytest.raises(ValueError, match="Some error"):
                plot_controller.param.trigger("plot_initiator")
            assert len(plot_controller.view) == 0
            assert not plot_controller.plot_generated
            assert plot_controller.plot_status == "Plot generation failed: Some error"

    def test_update_variable_param_choices(self):
        """
        Unit test for the _update_variable_param_choices method.

        The method is triggered indirectly by changing the 'plot_type' parameter.
        """
        ds_in = generate_dataset(data_var="temperature", frequency="monthly")
        plot_controller = app_classes_methods._PlotController(ds_in=ds_in)

        assert plot_controller.param.temporal_scope.objects == ["yearly", "monthly"]
        assert plot_controller.temporal_scope == "yearly"
        assert plot_controller.plot_type == "time series"

        plot_controller.plot_type = "map"  # triggers _update_variable_param_choices
        assert plot_controller.param.temporal_scope.objects == [
            "yearly",
            "monthly",
            "difference between years",
        ]

        plot_controller.temporal_scope = "difference between years"
        plot_controller.plot_type = "variance decomposition"
        assert plot_controller.param.temporal_scope.objects == ["yearly", "monthly"]
        assert plot_controller.temporal_scope == "yearly"

    def test_update_precedence(self):
        """
        Unit test for the _update_precedence method.

        The method is triggered indirectly by changing the 'plot_type' parameter.
        """
        ds_in = generate_dataset(data_var="temperature", frequency="monthly")
        plot_controller = app_classes_methods._PlotController(ds_in=ds_in)

        assert plot_controller.plot_type == "time series"
        assert plot_controller.param.location_string.precedence == 1
        assert plot_controller.param.location_selection.precedence == -1
        assert plot_controller.param.ensemble_stat.precedence == -1

        plot_controller.plot_type = "map"  # triggers _update_precedence

        assert plot_controller.param.location_string.precedence == -1
        assert plot_controller.param.location_selection.precedence == -1
        assert plot_controller.param.ensemble_stat.precedence == 1

    def test_revert_plot_status(self):
        """
        Unit test for the _revert_plot_status method.

        The method is triggered indirectly by changing the 'data_var' parameter.
        """
        ds_in = generate_dataset(
            data_var=["temperature", "precipitation"]
        ).climepi.sel_geo("Gabba")
        plot_controller = app_classes_methods._PlotController(ds_in=ds_in)
        plot_controller.param.trigger("plot_initiator")

        assert plot_controller.plot_generated
        assert plot_controller.plot_status == "Plot generated"

        plot_controller.data_var = "precipitation"  # triggers _revert_plot_status
        assert not plot_controller.plot_generated
        assert plot_controller.plot_status == "Plot not yet generated"


class TestController:
    """Unit tests for the Controller class."""

    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.climdata.EXAMPLES",
        {"data1": {"doc": "data1_doc"}, "data2": {"doc": "data2_doc"}},
    )
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model1": {"doc": "model1_doc"}, "model2": {"doc": "model2_doc"}},
    )
    def test_init(self, mock_get_example_model):
        """Unit test for the __init__ method."""

        def _mock_get_example_model(example_name):
            return epimod.SuitabilityModel(temperature_range=(1, 2))

        mock_get_example_model.side_effect = _mock_get_example_model

        controller = app_classes_methods.Controller(
            clim_dataset_example_base_dir="some/dir",
            clim_dataset_example_names=["data1", "data2"],
            enable_custom_clim_dataset=True,
            epi_model_example_names=["model1", "model2"],
            enable_custom_epi_model=True,
            custom_clim_data_dir="another/dir",
        )
        for attr, value in [
            ("clim_data_option", "Example dataset"),
            ("clim_example_name", "data1"),
            ("clim_example_doc", "data1_doc"),
            ("clim_data_load_initiator", False),
            ("clim_data_loaded", False),
            ("clim_data_status", "Data not loaded"),
            ("epi_model_option", "Example model"),
            ("epi_example_name", "model1"),
            ("epi_example_doc", "model1_doc"),
            ("epi_temperature_range", (15, 30)),
            ("epi_output_choice", "Suitable portion of each year"),
            ("suitability_threshold", 0),
            ("epi_model_run_initiator", False),
            ("epi_model_ran", False),
            ("epi_model_status", "Model has not been run"),
            ("_clim_dataset_example_base_dir", "some/dir"),
            ("_custom_clim_data_dir", "another/dir"),
            ("_ds_clim", None),
            ("_ds_epi", None),
        ]:
            assert getattr(controller, attr) == value, (
                f"Unexpected value for {attr}: expected {value}, "
                f"got {getattr(controller, attr)}"
            )
        assert controller.param.clim_data_option.precedence == 1
        assert controller._ds_epi_path.parents[1] == pathlib.Path(tempfile.gettempdir())
        assert controller._ds_epi_path.name == "ds_epi.nc"
        assert controller._epi_model.temperature_range == (1, 2)
        assert controller.param.epi_model_option.precedence == 1
        assert controller.param.suitability_threshold.precedence == -1
        assert isinstance(
            controller.clim_plot_controller, app_classes_methods._PlotController
        )
        assert controller.clim_plot_controller._ds_base is None
        assert isinstance(
            controller.epi_plot_controller, app_classes_methods._PlotController
        )
        assert controller.epi_plot_controller._ds_base is None
        assert isinstance(controller.data_controls, pn.Param)

    def test_clim_plot_controls(self):
        """
        Unit test for the clim_plot_controls method.

        Focuses on whether the controls are updated when expected when passed to a
        Panel app.
        """
        controller = app_classes_methods.Controller()
        controller.clim_plot_controller.controls = "some controls"
        # Create a panel object as would be done in the app
        controls_panel = pn.panel(controller.clim_plot_controls)
        assert controls_panel._pane.object == "some controls"
        # Updating any parameter in controller should update the controls panel
        controller.clim_plot_controller.controls = "new controls"
        assert controls_panel._pane.object == "some controls"
        controller.clim_data_loaded = True
        assert controls_panel._pane.object == "new controls"

    def test_clim_plot_view(self):
        """
        Unit test for the clim_plot_view method.

        Focuses on whether the view is updated when expected when passed to a Panel app.
        """
        controller = app_classes_methods.Controller()
        controller.clim_plot_controller.view = "some view"
        # Create a panel object as would be done in the app
        view_panel = pn.panel(controller.clim_plot_view)
        assert view_panel._pane.object == "some view"
        # Updating clim_plot_controller.view should only update the view panel when the
        # view_refresher event is triggered
        controller.clim_plot_controller.view = "new view"
        assert view_panel._pane.object == "some view"
        controller.clim_plot_controller.param.trigger("view_refresher")
        assert view_panel._pane.object == "new view"

    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model1": {}, "model2": {}, "model3": {}},
    )
    def test_epi_model_plot_view(self, mock_get_example_model):
        """
        Unit test for the epi_model_plot_view method.

        Tests that the view is updated when expected when passed to a Panel app.
        """
        model1 = epimod.SuitabilityModel(temperature_range=(1, 2))
        model2 = epimod.SuitabilityModel(temperature_range=(3, 4))
        model3 = "not a supported model"

        def _mock_get_example_model(example_name):
            if example_name == "model1":
                return model1
            if example_name == "model2":
                return model2
            if example_name == "model3":
                return model3
            raise ValueError(f"Unexpected example_name: {example_name}")

        mock_get_example_model.side_effect = _mock_get_example_model

        controller = app_classes_methods.Controller(
            epi_model_example_names=["model1", "model2", "model3"]
        )
        # Create a panel object as would be done in the app
        view_panel = pn.panel(controller.epi_model_plot_view)
        hvt.assertEqual(view_panel._pane[0].object, model1.plot_suitability())
        # Updating epi_model_name should update the view panel by triggering
        # _get_epi_model
        controller.epi_example_name = "model2"
        hvt.assertEqual(view_panel._pane[0].object, model2.plot_suitability())
        # Check case where epi_model.plot_suitability raises an error
        controller.epi_output_choice = "Suitability values"
        controller.epi_example_name = "model3"
        assert (
            view_panel._pane[0].object
            == "Error generating plot: 'str' object has no attribute "
            "'plot_suitability'"
        )

    def test_epi_plot_controls(self):
        """
        Unit test for the epi_plot_controls method.

        Focuses on whether the controls are updated when expected when passed to a
        Panel app.
        """
        controller = app_classes_methods.Controller()
        controller.epi_plot_controller.controls = "some controls"
        # Create a panel object as would be done in the app
        controls_panel = pn.panel(controller.epi_plot_controls)
        assert controls_panel._pane.object == "some controls"
        # Updating any parameter in controller should update the controls panel
        controller.epi_plot_controller.controls = "new controls"
        assert controls_panel._pane.object == "some controls"
        controller.epi_model_ran = True
        assert controls_panel._pane.object == "new controls"

    def test_epi_plot_view(self):
        """
        Unit test for the epi_plot_view method.

        Focuses on whether the view is updated when expected when passed to a Panel app.
        """
        controller = app_classes_methods.Controller()
        controller.epi_plot_controller.view = "some view"
        # Create a panel object as would be done in the app
        view_panel = pn.panel(controller.epi_plot_view)
        assert view_panel._pane.object == "some view"
        # Updating epi_plot_controller.view should only update the view panel (as used
        # in the app) when the view_refresher event is triggered
        controller.epi_plot_controller.view = "new view"
        assert view_panel._pane.object == "some view"
        controller.epi_plot_controller.param.trigger("view_refresher")
        assert view_panel._pane.object == "new view"

    @patch(
        "climepi.app._app_classes_methods.climdata.get_example_dataset",
        autospec=True,
    )
    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.climdata.EXAMPLES",
        {"data1": {}, "data2": {}, "data3": {}},
    )
    @patch.dict("climepi.app._app_classes_methods.epimod.EXAMPLES", {"model": {}})
    def test_load_clim_data(self, mock_get_example_model, mock_get_example_dataset):
        """
        Unit test for the _load_clim_data method.

        The method is triggered via the "clim_data_load_initiator" event parameter.
        """
        # Mock methods
        ds1 = generate_dataset(
            data_var="temperature", frequency="monthly", extra_dims={"realization": 2}
        ).climepi.sel_geo("SCG")
        ds2 = generate_dataset(
            data_var="precipitation", frequency="daily", extra_dims={"realization": 2}
        ).climepi.sel_geo("Lords")

        def _mock_get_example_dataset(clim_example_name, base_dir=None):
            assert str(base_dir) == "some/dir"
            if clim_example_name == "data1":
                return ds1
            if clim_example_name == "data2":
                return ds2
            raise ValueError("Dataset not available")

        mock_get_example_dataset.side_effect = _mock_get_example_dataset

        epi_model = epimod.SuitabilityModel(temperature_range=(1, 2))
        mock_get_example_model.return_value = epi_model

        # Create controller

        controller = app_classes_methods.Controller(
            clim_dataset_example_base_dir="some/dir",
            clim_dataset_example_names=["data1", "data2", "data3"],
            epi_model_example_names=["model"],
            enable_custom_clim_dataset=True,
            custom_clim_data_dir="another/dir",
        )

        # Test loading the first dataset

        assert controller.clim_example_name == "data1"  # should be default
        assert not controller.clim_data_loaded
        assert controller.clim_data_status == "Data not loaded"
        controller.param.trigger("clim_data_load_initiator")

        assert controller.clim_data_loaded
        assert controller.clim_data_status == "Data loaded"
        xrt.assert_identical(controller._ds_clim, ds1)
        xrt.assert_identical(controller.clim_plot_controller._ds_base, ds1)
        assert controller.clim_plot_controller.plot_status == "Plot not yet generated"

        # Run an epi model and generate some plots to test loading a different dataset
        # resets things as expected

        controller.param.trigger("epi_model_run_initiator")
        assert controller.epi_plot_controller._ds_base is not None
        controller.clim_plot_controller.param.trigger("plot_initiator")
        controller.epi_plot_controller.param.trigger("plot_initiator")
        assert controller.clim_plot_controller.plot_generated
        assert controller.epi_plot_controller.plot_generated

        # Test triggering the clim_data_load_initiator event again does not reload the
        # data
        with patch(
            "climepi.app._app_classes_methods._load_clim_data_func",
            side_effect=ValueError,
        ):
            controller.param.trigger("clim_data_load_initiator")
        assert controller.clim_data_loaded
        assert controller.clim_data_status == "Data loaded"
        xrt.assert_identical(controller._ds_clim, ds1)

        # Test loading a different dataset

        controller.clim_example_name = "data2"
        assert not controller.clim_data_loaded
        assert controller.clim_data_status == "Data not loaded"

        controller.param.trigger("clim_data_load_initiator")

        assert controller.clim_data_loaded
        assert controller.clim_data_status == "Data loaded"
        xrt.assert_identical(controller._ds_clim, ds2)
        xrt.assert_identical(controller.clim_plot_controller._ds_base, ds2)
        assert controller.epi_plot_controller._ds_base is None
        assert not controller.clim_plot_controller.plot_generated
        assert not controller.epi_plot_controller.plot_generated

        # Test loading a custom dataset

        controller.clim_data_option = "Custom dataset"
        assert not controller.clim_data_loaded
        assert controller.clim_data_status == "Data not loaded"

        with patch(
            "climepi.app._app_classes_methods._load_clim_data_func",
            autospec=True,
            return_value=ds1,
        ) as mock_load_clim_data_func:
            controller.param.trigger("clim_data_load_initiator")

        mock_load_clim_data_func.assert_called_once_with(
            "Custom dataset",
            clim_example_name="data2",
            clim_example_base_dir="some/dir",
            custom_clim_data_dir="another/dir",
        )
        assert controller.clim_data_loaded
        assert controller.clim_data_status == "Data loaded"
        xrt.assert_identical(controller._ds_clim, ds1)
        xrt.assert_identical(controller.clim_plot_controller._ds_base, ds1)

        # Check error handling
        controller.clim_data_option = "Example dataset"
        controller.clim_example_name = "data3"
        with pytest.raises(ValueError, match="Dataset not available"):
            controller.param.trigger("clim_data_load_initiator")
        assert not controller.clim_data_loaded
        assert controller.clim_data_status == "Data load failed: Dataset not available"

        # Run cleanup on temp file
        controller.cleanup_temp_file()

    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model1": {}, "model2": {}, "model3": {}},
    )
    def test_get_epi_model(self, mock_get_example_model):
        """
        Unit test for the _get_epi_model method.

        The method is triggered via changing any of the "epi_model_option",
        "epi_example_name", or "epi_temperature_range" parameters.
        """
        temperature_range1 = (1, 2)
        suitability_table2 = xr.Dataset(
            {"suitability": ("temperature", [0.1, 0.2])},
            coords={"temperature": [15, 30]},
        )

        def _mock_get_example_model(example_name):
            if example_name == "model1":
                return epimod.SuitabilityModel(temperature_range=temperature_range1)
            if example_name == "model2":
                return epimod.SuitabilityModel(suitability_table=suitability_table2)
            raise ValueError("Some error")

        mock_get_example_model.side_effect = _mock_get_example_model

        controller = app_classes_methods.Controller(
            epi_model_example_names=["model1", "model2", "model3"],
            enable_custom_epi_model=True,
        )

        # model1 should be used by default
        assert controller.epi_model_option == "Example model"
        assert controller.epi_example_name == "model1"
        assert controller.param.epi_example_name.precedence == 1
        assert controller.param.epi_temperature_range.precedence == -1
        assert controller._epi_model.temperature_range == temperature_range1
        assert controller._epi_model.suitability_table is None
        assert controller.param.suitability_threshold.precedence == -1

        # Change to model2
        controller.epi_example_name = "model2"

        assert controller._epi_model.temperature_range is None
        xrt.assert_equal(controller._epi_model.suitability_table, suitability_table2)
        assert controller.param.suitability_threshold.precedence == 1
        assert controller.param.suitability_threshold.bounds == (0, 0.2)

        # Change to model that raises an error
        with pytest.raises(ValueError, match="Some error"):
            controller.epi_example_name = "model3"
        assert (
            controller.epi_model_status
            == "Error getting epidemiological model: Some error"
        )
        assert controller._epi_model is None

        # Change to a custom model
        controller.epi_model_option = "Custom temperature-dependent suitability model"
        # controller.epi_temperature_range = (10, 20)
        assert controller.epi_temperature_range == (15, 30)
        assert controller.param.epi_example_name.precedence == -1
        assert controller.param.epi_temperature_range.precedence == 1
        assert controller._epi_model.temperature_range == (15, 30)
        assert controller._epi_model.suitability_table is None
        assert controller.param.suitability_threshold.precedence == -1

        # Change to another custom model
        controller.epi_temperature_range = (10, 20)
        assert controller._epi_model.temperature_range == (10, 20)

        # Provide an unsupported epi_model_option value
        controller.param.epi_model_option.objects.append("Some unsupported option")
        with pytest.raises(
            ValueError,
            match="Unrecognised epidemiological model option: Some unsupported option",
        ):
            controller.epi_model_option = "Some unsupported option"
        assert (
            controller.epi_model_status
            == "Unrecognised epidemiological model option: Some unsupported option"
        )

    @patch(
        "climepi.app._app_classes_methods.climdata.get_example_dataset",
        autospec=True,
    )
    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.climdata.EXAMPLES",
        {"data": {}},
    )
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model1": {}, "model2": {}, "model3": {}},
    )
    def test_run_epi_model(self, mock_get_example_model, mock_get_example_dataset):
        """
        Unit test for the _load_clim_data method.

        The method is triggered via the "epi_model_run_initiator" event parameter.
        """
        ds = generate_dataset(
            data_var="temperature", frequency="monthly", extra_dims={"realization": 2}
        ).climepi.sel_geo("SCG")
        mock_get_example_dataset.return_value = ds

        epi_model1 = epimod.SuitabilityModel(temperature_range=(0, 0.5))
        epi_model2 = epimod.SuitabilityModel(
            suitability_table=xr.Dataset(
                {"suitability": ("temperature", [0.1, 0.9, 0.3])},
                coords={"temperature": [0.25, 0.5, 0.75]},
            )
        )
        epi_model3 = "Not a supported model"

        def _mock_get_example_model(example_name):
            if example_name == "model1":
                return epi_model1
            if example_name == "model2":
                return epi_model2
            if example_name == "model3":
                return epi_model3
            raise ValueError(f"Unexpected example_name: {example_name}")

        mock_get_example_model.side_effect = _mock_get_example_model

        controller = app_classes_methods.Controller(
            clim_dataset_example_names=["data"],
            epi_model_example_names=["model1", "model2", "model3"],
        )

        # First try without climate data loaded
        controller.param.trigger("epi_model_run_initiator")
        assert not controller.epi_model_ran
        assert controller.epi_model_status == "Need to load climate data"
        assert controller._ds_epi is None
        assert controller.epi_plot_controller._ds_base is None

        # Load climate data
        controller.param.trigger("clim_data_load_initiator")
        assert controller.clim_data_loaded

        # Check that triggering with epi_model_ran = True does not run the model (this
        # is mainly to prevent re-running but for simplicity of testing we just check
        # that it isn't run at all if not actually run before)
        controller.epi_model_ran = True
        controller.param.trigger("epi_model_run_initiator")
        assert controller._ds_epi is None
        assert controller.epi_plot_controller._ds_base is None

        # Run model1, returning raw suitability values
        controller.epi_model_ran = False
        assert controller.epi_example_name == "model1"
        controller.epi_output_choice = "Suitability values"
        controller.param.trigger("epi_model_run_initiator")
        assert controller.epi_model_ran
        assert controller.epi_model_status == "Model run complete"
        xrt.assert_identical(
            controller._ds_epi, controller.epi_plot_controller._ds_base
        )
        xrt.assert_allclose(
            controller._ds_epi,
            epi_model1.run(ds, return_yearly_portion_suitable=False),
        )

        # Run model2, returning yearly portion suitable
        controller.epi_example_name = "model2"
        controller.epi_output_choice = "Suitable portion of each year"
        controller.suitability_threshold = 0.5
        assert not controller.epi_model_ran
        assert controller.epi_model_status == "Model has not been run"
        controller.param.trigger("epi_model_run_initiator")
        assert controller.epi_model_ran
        assert controller.epi_model_status == "Model run complete"
        xrt.assert_allclose(
            controller._ds_epi,
            epi_model2.run(
                ds, return_yearly_portion_suitable=True, suitability_threshold=0.5
            ),
        )
        xrt.assert_identical(
            controller._ds_epi, controller.epi_plot_controller._ds_base
        )

        # Check error handling
        controller.epi_output_choice = "Suitability values"
        controller.epi_example_name = "model3"
        with pytest.raises(AttributeError, match="'str' object has no attribute 'run'"):
            controller.param.trigger("epi_model_run_initiator")
        assert not controller.epi_model_ran
        assert (
            controller.epi_model_status
            == "Model run failed: 'str' object has no attribute 'run'"
        )

        controller.epi_example_name = "model1"  # check a valid model can be run again
        controller.epi_output_choice = "Suitable portion of each year"
        controller.param.trigger("epi_model_run_initiator")
        assert controller.epi_model_ran
        assert controller.epi_model_status == "Model run complete"
        xrt.assert_allclose(
            controller._ds_epi,
            epi_model1.run(
                ds, return_yearly_portion_suitable=True, suitability_threshold=0
            ),
        )

        controller.epi_model_ran = False
        controller._epi_model = None  # mimic case where model not obtained properly
        controller.param.trigger("epi_model_run_initiator")
        assert not controller.epi_model_ran
        assert (
            controller.epi_model_status
            == "Need to select a valid epidemiological model"
        )

        # Run cleanup on temp file
        controller.cleanup_temp_file()

    @patch.dict(
        "climepi.app._app_classes_methods.climdata.EXAMPLES",
        {"data1": {}, "data2": {}},
    )
    def test_revert_clim_data_load_status(self):
        """
        Unit test for the _revert_clim_data_load_status method.

        The method is triggered by changing the 'clim_example_name' parameter.
        """
        controller = app_classes_methods.Controller(
            clim_dataset_example_names=["data1", "data2"]
        )
        assert controller.clim_example_name == "data1"
        controller.clim_data_loaded = True
        controller.clim_data_status = "Data loaded"
        controller.clim_example_name = "data2"
        assert not controller.clim_data_loaded
        assert controller.clim_data_status == "Data not loaded"
        # Check that reselecting the current dataset does not revert the status
        controller.clim_data_loaded = True
        controller.clim_data_status = "Data loaded"
        controller.clim_example_name = "data2"
        assert controller.clim_data_loaded
        assert controller.clim_data_status == "Data loaded"

    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.climdata.EXAMPLES",
        {"data1": {}, "data2": {}},
    )
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model1": {}, "model2": {}},
    )
    def test_revert_epi_model_run_status(self, _):
        """
        Unit test for the _revert_epi_model_run_status method.

        The method is triggered by changing the 'clim_example_name', 'epi_model_option',
        'epi_example_name', 'epi_temperature_range', 'epi_output_choice', or
        'suitability_threshold' parameters.
        """
        controller = app_classes_methods.Controller(
            clim_dataset_example_names=["data1", "data2"],
            epi_model_example_names=["model1", "model2"],
        )
        assert controller.clim_example_name == "data1"
        assert controller.epi_model_option == "Example model"
        assert controller.epi_example_name == "model1"
        assert controller.epi_temperature_range == (15, 30)
        assert controller.epi_output_choice == "Suitable portion of each year"
        assert controller.suitability_threshold == 0
        assert not controller.epi_model_ran
        assert controller.epi_model_status == "Model has not been run"
        for attr, new_value in [
            ("clim_example_name", "data2"),
            ("epi_example_name", "model2"),
            ("epi_model_option", "Custom temperature-dependent suitability model"),
            ("epi_temperature_range", (10, 20)),
            ("epi_output_choice", "Suitability values"),
            ("suitability_threshold", 0.5),
        ]:
            controller.epi_model_ran = True
            controller.epi_model_status = "Model run complete"
            setattr(controller, attr, new_value)
            assert not controller.epi_model_ran
            assert controller.epi_model_status == "Model has not been run"

    @patch.dict(
        "climepi.app._app_classes_methods.climdata.EXAMPLES",
        {"data1": {"doc": "doc1"}, "data2": {"doc": "doc2"}, "data3": {}},
    )
    def test_update_clim_example_name_doc(self):
        """
        Unit test for the _update_clim_example_name_doc method.

        The method is triggered by changing the 'clim_example_name' parameter.
        """
        controller = app_classes_methods.Controller(
            clim_dataset_example_names=["data1", "data2"],
            enable_custom_clim_dataset=True,
        )
        assert controller.clim_example_doc == "doc1"
        assert controller.param.clim_example_name.precedence == 1
        assert controller.param.clim_example_doc.precedence == 1
        # Change to another dataset with doc string
        controller.clim_example_name = "data2"
        assert controller.clim_example_doc == "doc2"
        assert controller.param.clim_example_name.precedence == 1
        assert controller.param.clim_example_doc.precedence == 1
        # Change to dataset without doc string
        controller.clim_example_name = "data3"
        assert controller.clim_example_doc == ""
        assert controller.param.clim_example_name.precedence == 1
        assert controller.param.clim_example_doc.precedence == 1
        # Change to custom dataset
        controller.clim_data_option = "Custom dataset"
        assert controller.param.clim_example_name.precedence == -1
        assert controller.param.clim_example_doc.precedence == -1
        # Change back to example dataset
        controller.clim_data_option = "Example dataset"
        assert controller.clim_example_name == "data3"
        assert controller.clim_example_doc == ""
        assert controller.param.clim_example_name.precedence == 1
        assert controller.param.clim_example_doc.precedence == 1
        # Change to unsupported option (n.b. maybe this catch not needed as param
        # already notices when we try to change clim_data_option to an unsupported
        # value)
        controller.param.clim_data_option.objects.append("Short of a length")
        with pytest.raises(
            ValueError, match="Unrecognised climate data option: Short of a length"
        ):
            controller.clim_data_option = "Short of a length"

    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model1": {"doc": "doc1"}, "model2": {}},
    )
    def test_update_epi_model_doc(self, _):
        """
        Unit test for the _update_epi_model_doc method.

        The method is triggered by changing the 'epi_example_name' and
        'epi_model_option' parameters.
        """
        controller = app_classes_methods.Controller(
            epi_model_example_names=["model1", "model2"]
        )
        assert controller.epi_example_doc == "doc1"
        assert controller.param.epi_example_doc.precedence == 1
        controller.epi_example_name = "model2"
        assert controller.epi_example_doc == ""
        assert controller.param.epi_example_doc.precedence == 1
        controller.epi_model_option = "Custom temperature-dependent suitability model"
        assert controller.param.epi_example_doc.precedence == -1
        # Test changing epi_example_name when an example model is not actually being
        # used (maybe unnecessary as this shouldn't actually be possible in the app)
        controller.epi_example_name = "model1"
        assert controller.epi_example_doc == ""
        assert controller.param.epi_example_doc.precedence == -1
        controller.epi_model_option = "Example model"
        assert controller.epi_example_doc == "doc1"
        assert controller.param.epi_example_doc.precedence == 1

    def test_update_epi_example_model_temperature_range_precedence(self):
        """
        Unit test for the _update_epi_example_model_temperature_range_precedence method.

        The method is triggered by changing the 'epi_model_option' parameter.
        """
        controller = app_classes_methods.Controller()
        assert controller.epi_model_option == "Example model"
        assert controller.param.epi_example_name.precedence == 1
        assert controller.param.epi_temperature_range.precedence == -1
        controller.epi_model_option = "Custom temperature-dependent suitability model"
        assert controller.param.epi_example_name.precedence == -1
        assert controller.param.epi_temperature_range.precedence == 1
        controller.epi_model_option = "Example model"
        assert controller.param.epi_example_name.precedence == 1
        assert controller.param.epi_temperature_range.precedence == -1
        controller.param.epi_model_option.objects.append("Some unsupported option")
        with pytest.raises(ValueError):
            # Error actually first raised by _get_epi_model when trying an unsupported
            # option
            controller.epi_model_option = "Some unsupported option"
        with pytest.raises(
            ValueError,
            match="Unrecognised epidemiological model option: Some unsupported option",
        ):
            controller._update_epi_example_model_temperature_range_precedence()
        assert controller.param.epi_example_name.precedence == 1
        assert controller.param.epi_temperature_range.precedence == -1

    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model1": {}, "model2": {}, "model3": {}},
    )
    def test_update_suitability_threshold(self, mock_get_example_model):
        """
        Unit test for the _update_suitability_threshold method.

        The method is triggered by changing the 'epi_output_choice' parameter.
        """
        # Note suitability is binary for models 1 and 3, so threshold precedence should
        # only be positive for model 2
        epi_model1 = epimod.SuitabilityModel(temperature_range=(0, 0.5))
        epi_model2 = epimod.SuitabilityModel(
            suitability_table=xr.Dataset(
                {"suitability": ("temperature", [0.1, 0.9, 0.3])},
                coords={"temperature": [0.25, 0.5, 0.75]},
            )
        )
        epi_model3 = epimod.SuitabilityModel(
            suitability_table=xr.Dataset(
                {"suitability": ("temperature", [False, True, False])},
                coords={"temperature": [0.25, 0.5, 0.75]},
            )
        )

        def _mock_get_example_model(example_name):
            if example_name == "model1":
                return epi_model1
            if example_name == "model2":
                return epi_model2
            if example_name == "model3":
                return epi_model3
            raise ValueError(f"Unexpected example_name: {example_name}")

        mock_get_example_model.side_effect = _mock_get_example_model

        controller = app_classes_methods.Controller(
            epi_model_example_names=["model1", "model2", "model3"]
        )
        assert controller.epi_example_name == "model1"
        assert controller.epi_output_choice == "Suitable portion of each year"
        assert controller.suitability_threshold == 0
        assert controller.param.suitability_threshold.precedence == -1

        controller.epi_example_name = "model2"
        assert controller.suitability_threshold == 0
        assert controller.param.suitability_threshold.precedence == 1
        assert controller.param.suitability_threshold.bounds == (0, 0.9)

        controller.epi_output_choice = "Suitability values"
        assert controller.param.suitability_threshold.precedence == -1
        controller.epi_output_choice = "Suitable portion of each year"
        assert controller.param.suitability_threshold.precedence == 1

        controller.epi_example_name = "model3"
        assert controller.param.suitability_threshold.precedence == -1

        controller.param.epi_output_choice.objects.append("Some unsupported option")
        with pytest.raises(
            ValueError, match="Unrecognised epidemiological model output choice"
        ):
            controller.epi_output_choice = "Some unsupported option"

    @patch(
        "climepi.app._app_classes_methods.climdata.get_example_dataset",
        autospec=True,
    )
    @patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
    @patch.dict(
        "climepi.app._app_classes_methods.climdata.EXAMPLES",
        {"data": {}},
    )
    @patch.dict(
        "climepi.app._app_classes_methods.epimod.EXAMPLES",
        {"model": {}},
    )
    def test_cleanup_temp_file(self, mock_get_example_model, mock_get_example_dataset):
        """Unit test for the cleanup_temp_file method."""
        ds = generate_dataset(
            data_var="temperature", frequency="monthly", extra_dims={"realization": 2}
        ).climepi.sel_geo("SCG")
        mock_get_example_dataset.return_value = ds

        epi_model = epimod.SuitabilityModel(temperature_range=(0, 0.5))
        mock_get_example_model.return_value = epi_model

        controller = app_classes_methods.Controller(
            clim_dataset_example_names=["data"], epi_model_example_names=["model"]
        )

        # Load climate data and run epi model to create temp file
        controller.param.trigger("clim_data_load_initiator")
        controller.param.trigger("epi_model_run_initiator")

        # Generate plots using dataset from temp file to ensure this doesn't interfere
        # with cleanup
        controller.epi_plot_controller.param.trigger("plot_initiator")
        assert isinstance(controller.epi_plot_view()[0][1].object, hv.Overlay)

        assert controller._ds_epi is not None
        assert controller._ds_epi_path.exists()
        controller.cleanup_temp_file()
        assert not controller._ds_epi_path.exists()
        assert not controller._ds_epi_path.parent.exists()
