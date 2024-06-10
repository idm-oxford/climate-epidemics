from climepi.app import _run_app

app = _run_app._get_app(clim_dataset_example_names=["isimip_london"]).servable()
