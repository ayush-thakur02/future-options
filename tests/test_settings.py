"""Configuration loading, including the per-plugin YAML tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import load_settings
from kernel.loader import discover


def test_loads_nested_plugin_yaml_by_handle(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    plugin_dir = config_dir / "plugins" / "forecast"
    plugin_dir.mkdir(parents=True)
    config_path = config_dir / "default.yaml"
    config_path.write_text("model:\n  horizons: [1, 3]\n")
    (plugin_dir / "projection.yaml").write_text("bars_ahead: 7\n")

    settings = load_settings(config_path)

    assert settings.horizons == (1, 3)
    assert settings.plugin_config == {"forecast:projection": {"bars_ahead": 7}}


def test_inline_plugin_block_remains_a_higher_priority_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    plugin_dir = config_dir / "plugins" / "forecast"
    plugin_dir.mkdir(parents=True)
    config_path = config_dir / "default.yaml"
    config_path.write_text(
        "plugins:\n"
        "  forecast:projection:\n"
        "    bars_ahead: 9\n"
    )
    (plugin_dir / "projection.yaml").write_text(
        "bars_ahead: 3\n"
        "refresh_on_close: true\n"
    )

    settings = load_settings(config_path)

    assert settings.plugin_config["forecast:projection"] == {
        "bars_ahead": 9,
        "refresh_on_close": True,
    }


def test_plugin_yaml_must_be_a_mapping(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    plugin_dir = config_dir / "plugins" / "source"
    plugin_dir.mkdir(parents=True)
    config_path = config_dir / "default.yaml"
    config_path.write_text("{}\n")
    (plugin_dir / "simulated.yaml").write_text("- seed\n- days\n")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_settings(config_path)


def test_every_bundled_plugin_has_a_config_file() -> None:
    project = Path(__file__).resolve().parents[1]
    settings = load_settings(project / "config" / "default.yaml")
    bundled_handles = {entry.handle for entry in discover().entries}

    assert set(settings.plugin_config) == bundled_handles
