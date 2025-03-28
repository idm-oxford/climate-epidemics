def _get_data_version():
    version = pooch.check_version(get_versions()["version"], fallback="main")
    if version != "main":
        version = "v" + version
    return version
