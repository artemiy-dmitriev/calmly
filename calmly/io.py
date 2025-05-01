import yaml

def load_config(config_path: str) -> dict:
    """
    Loads the project config from a yaml file.

    Parameters
    ----------
    config_path : str
        Path to the yaml file that contains settings for a `calmly` project. Default is `calmly_config.yaml`.

    Returns
    -------
    config : dict
        A Python dictionary containing the project settings
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)