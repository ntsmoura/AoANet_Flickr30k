from pathlib import Path

from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[".secrets.toml"],
    root_path=Path(__file__).parent,
    silent=False,
    merge_enabled=True,
)