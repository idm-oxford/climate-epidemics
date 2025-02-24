"""Unit tests for the _app_classes_methods module of the app subpackage."""

import pathlib
import tempfile
from unittest.mock import patch

import holoviews as hv
import numpy as np
import numpy.testing as npt
import panel as pn
import param
import pytest
import xarray.testing as xrt
from holoviews.element.comparison import Comparison as hvt

import climepi  # noqa
import climepi.app._app_classes_methods as app_classes_methods
from climepi import epimod
from climepi.testing.fixtures import generate_dataset

original_plot_map = climepi.ClimEpiDatasetAccessor.plot_map


def _plot_map(self, *args, **kwargs):
    """
    Run dataset.climepi.plot_map method but insisting rasterize=False.

    This is needed because the default rasterize=True option seems to cause an error
    when in debug mode.
    """
    return original_plot_map(self, *args, **{**kwargs, "rasterize": False})


@patch("climepi.app._app_classes_methods.climdata.get_example_dataset", autospec=True)
def test_load_clim_data_func(mock_get_example_dataset):
    """Unit test for the _load_clim_data_func function."""
    mock_get_example_dataset.return_value = "mocked_dataset"
    result = app_classes_methods._load_clim_data_func("some_example_name", "some/dir")
    mock_get_example_dataset.assert_called_once_with(
        "some_example_name", base_dir="some/dir"
    )
    assert result == "mocked_dataset"
    # Check cached version is returned if the same example_name and base_dir are
    # provided
    mock_get_example_dataset.return_value = "another_mocked_dataset"
    result_cached = app_classes_methods._load_clim_data_func(
        "some_example_name", "some/dir"
    )
    assert result_cached == "mocked_dataset"
    mock_get_example_dataset.assert_called_once()


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
            with pytest.raises(ValueError, match="Unsupported"):
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
                ("Ocean", "I"),
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
        "temporal_scope_base,spatial_scope_base,scenario_scope_base,model_scope_base,ensemble_scope_base",
        [
            ("daily", "list", "single", "single", "single"),
            ("monthly", "grid", "multiple", "multiple", "multiple"),
            ("yearly", "list", "single", "single", "multiple"),
            ("fake option", "list", "single", "single", "single"),
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
        ds_in = generate_dataset(data_var="temperature").climepi.sel_geo("SCG")
        plot_controller = app_classes_methods._PlotController(ds_in=ds_in)
        plot_controller.view.append("some_view")

        view_refresher_trigger_count = 0

        @param.depends("plot_controller.view_refresher")
        def _update_view_refresher_triggers():
            nonlocal view_refresher_trigger_count
            view_refresher_trigger_count += 1

        plot_controller.param.trigger("plot_initiator")

        assert len(plot_controller.view) == 1
        assert plot_controller.plot_generated
        assert plot_controller.plot_status == "Plot generated"
        assert view_refresher_trigger_count == 2  # refreshed at start/end of generation

        # Check that the view is not updated if the plot has already been generated
        plot_controller.param.trigger("plot_initiator")
        assert view_refresher_trigger_count == 2

        # Check error handling
        plot_controller.initialize()
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
