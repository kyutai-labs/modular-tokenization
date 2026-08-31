"""Configuration parsing shared by all CLIs of the package.

Configs are pydantic models filled from the command line via OmegaConf dotted
overrides (``key.subkey=value``), optionally layered on top of YAML files given
with ``config=path.yaml`` (or a list of paths). Later sources override earlier
ones; CLI overrides everything.
"""
import copy
import logging
import sys
from typing import Any, Type, TypeVar

import omegaconf
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def as_str_list(value) -> list[str]:
    """Normalize a comma-separated value ("a,b,c") or any sequence of values
    to a list of strings."""
    if isinstance(value, str):
        return [v.strip() for v in value.split(",")]
    return [str(v).strip() for v in value]


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=(level or "INFO").upper(),
        stream=sys.stdout,
    )


def parse_file_config(path: str) -> DictConfig:
    file_cfg = OmegaConf.load(path)
    if not isinstance(file_cfg, DictConfig):
        raise ValueError(
            f"File paths must parse to DictConfig, but it was: {type(file_cfg)}"
        )
    return file_cfg


def recursively_parse_config(cfg: DictConfig) -> list[DictConfig]:
    if "config" not in cfg:
        return [cfg]

    ordered_cfgs = []
    cfg = copy.deepcopy(cfg)
    config_arg = cfg["config"]
    del cfg["config"]
    ordered_cfgs.append(cfg)

    if isinstance(config_arg, str):
        file_cfg = parse_file_config(config_arg)
        sub_configs = recursively_parse_config(file_cfg)
        ordered_cfgs = sub_configs + ordered_cfgs
    elif isinstance(config_arg, omegaconf.listconfig.ListConfig):
        sub_configs = []
        for c in config_arg:
            if not isinstance(c, str):
                raise ValueError(
                    f'If "config" is specified, it must be either a string path or a list of string paths. It was config={config_arg}'
                )
            config_to_parse = parse_file_config(c)
            sub_configs.extend(recursively_parse_config(config_to_parse))
        ordered_cfgs = sub_configs + ordered_cfgs
    else:
        raise ValueError(
            f'If "config" is specified, it must be either a string path or a list of string paths, it was config={config_arg}'
        )
    return ordered_cfgs


def parse_args_with_default(
    *,
    default_cfg: DictConfig | None = None,
    cli_args: DictConfig | None = None,
):
    """Merge default config, config files and CLI overrides (later wins)."""
    if cli_args is None:
        cli_args = OmegaConf.from_cli()
        assert isinstance(
            cli_args, DictConfig
        ), f"CLI Args must be a DictConfig, not {type(cli_args)}"
    ordered_cfgs = recursively_parse_config(cli_args)
    if default_cfg is not None:
        ordered_cfgs.insert(0, default_cfg)
    cfg = OmegaConf.merge(*ordered_cfgs)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


T = TypeVar("T", bound=BaseModel)


def get_pydantic_default_args(args_cls: Type[T]) -> dict[str, Any]:
    defaults = {}
    for field, info in args_cls.model_fields.items():
        if info.default != PydanticUndefined:
            defaults[field] = info.default
    return defaults


def parse_args_to_pydantic_model(
    args_cls: Type[T],
    cli_args: DictConfig | None = None,
    instantiate_default_cls: bool = True,
) -> T:
    if instantiate_default_cls:
        default_cfg = OmegaConf.create(args_cls().model_dump())
    else:
        default_cfg = OmegaConf.create(get_pydantic_default_args(args_cls))
    parsed_cfg = parse_args_with_default(default_cfg=default_cfg, cli_args=cli_args)
    pydantic_args = args_cls.model_validate(parsed_cfg)
    return pydantic_args
