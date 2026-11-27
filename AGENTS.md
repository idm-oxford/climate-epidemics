# AGENTS.md

climepi is a Python package and Panel web app that combines climate data with
climate-sensitive epidemiological models.

## Layout

- `climepi/_core.py`: the `.climepi` xarray accessor (temporal averaging, `sel_geo`,
  ensemble statistics, plotting).
- `climepi/climdata/`: climate data retrieval. Each source subclasses
  `ClimateDataGetter` (`_data_getter_class.py`) and is registered in `_base.py`.
  Example datasets are defined in `_examples.py`.
- `climepi/epimod/`: epidemiological models. `climepi/app/`: the web app.
- `tests/test_<subpackage>_<module>.py` mirrors the modules.

## Commands (pixi)

- `pixi run test`. Add `--no-cov` and `-k` when iterating. App tests need
  `pixi run playwright install chromium`.
- `pixi run lint`, `pixi run ruff format`, and `pixi run type-check` (mypy, `climepi/`
  only).
- `pixi run docs` builds the Sphinx docs.

## Conventions

- Ruff with line length 88 and NumPy-style docstrings. Public functions and classes get
  docstrings. Private methods open with a `#` comment instead.
- Type-hint new code and keep mypy clean.
- Use American spelling ("center", "behavior").
- Match the naming, structure and idioms of neighboring modules. Put logic shared by
  data sources in the base class or `climdata/_utils.py` instead of repeating it.
- In new tests, mock remote services with `unittest.mock`, build data with
  `climepi.testing.fixtures`, and compare with `xarray.testing`. Give every test a
  docstring, e.g. "Test the _process_data method of the ISIMIPDataGetter class."
- A new data source also needs the `get_climate_data` docstring and
  `docs/getting-started/overview.rst` updated.
- Do not hand-edit `pixi.lock`. A scheduled workflow updates it.
- Only commit data in the example directories whitelisted in `.gitignore`.
- Branch off `main` and open PRs against it. CI runs lint, mypy and tests on Linux,
  macOS and Windows.
