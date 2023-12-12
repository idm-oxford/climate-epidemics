"""Base module of the epimod subpackage providing a base epidemiological model
class and an xarray accessor class.
"""

import xarray as xr

import climepi  # noqa # pylint: disable=unused-import


class EpiModel:
    """Base class for epidemiological models."""

    def __init__(self):
        pass

    def run(self, ds_clim):
        """
        Runs the epidemiological model on a given climate dataset.

        Parameters:
        -----------
        ds_clim : xarray.Dataset
            The input climate dataset.

        Returns:
        --------
        xarray.Dataset
            The output epidemiological dataset.
        """
        da_epi = self.run_main(ds_clim)
        ds_epi = xr.Dataset(attrs=ds_clim.attrs)
        ds_epi[da_epi.name] = da_epi
        ds_epi.climepi.copy_bnds_from(ds_clim)
        ds_epi.climepi.modes = ds_clim.climepi.modes.copy()

        return ds_epi

    def run_main(self, ds_clim):
        """
        Abstract method that must be implemented by subclasses. Runs the main
        logic of the epidemiological model.

        Parameters:
        -----------
        ds_clim : xarray.Dataset
            The input climate dataset.
        """
        raise NotImplementedError


@xr.register_dataset_accessor("epimod")
class EpiModDatasetAccessor:
    """
    Accessor class for running epidemiological models on xarray climate
    datasets.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._model = None

    @property
    def model(self):
        """Gets and sets the epidemiological model (EpiModel object)."""
        if self._model is None:
            raise ValueError("Model not set.")
        return self._model

    @model.setter
    def model(self, model_in):
        if not isinstance(model_in, EpiModel):
            raise ValueError("Model must be an instance of EpiModel.")
        self._model = model_in

    def run_model(self):
        """
        Runs the epidemiological model on the climate dataset.

        Returns:
        -------
        xarray.Dataset:
            The output of the model's run method.
        """
        ds_epi = self.model.run(self._obj)
        return ds_epi

    def months_suitable(self):
        """
        Calculates the number of months suitable each year from monthly suitability
        data.

        Returns:
        --------
        xarray.Dataset:
            Dataset with a single non-bounds variable "months_suitable".
        """
        if self._obj.climepi.modes["temporal"] != "monthly":
            raise ValueError("Suitability data must be monthly.")
        try:
            ds_suitability = self._obj.climepi.sel_data_var("suitability")
        except KeyError as exc:
            raise KeyError(
                """No suitability data found. To calculate the number of months suitable
                from a climate dataset, first run the suitability model and then apply
                this method to the output dataset."""
            ) from exc
        ds_suitability_mean = ds_suitability.climepi.annual_mean(weighted=False)
        ds_months_suitable = 12 * ds_suitability_mean.rename(
            {"suitability": "months_suitable"}
        )
        ds_months_suitable.months_suitable.attrs.update(units="months")
        ds_months_suitable.climepi.modes = ds_suitability_mean.climepi.modes
        return ds_months_suitable
