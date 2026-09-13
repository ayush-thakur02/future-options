"""Tests for the plugin kernel.

The kernel is the load-bearing part of the architecture, so these tests cover
the properties the rest of the platform relies on rather than individual lines:
discovery finds nested plugins, a capability resolves to its single provider,
building is memoised, a bad plugin is reported instead of fatal, and a missing
capability is caught before anything runs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from niftypulse.core.settings import Settings
from niftypulse.kernel import (
    DuplicatePlugin,
    EventBus,
    Kernel,
    MissingDependency,
    NoProvider,
    PluginKind,
    PluginManifest,
    Topic,
    UnknownPlugin,
    parse_handle,
)
from niftypulse.kernel.context import PluginContext
from niftypulse.kernel.errors import PluginLoadError
from niftypulse.kernel.loader import PLUGIN_MODULE, load_module
from niftypulse.kernel.registry import PluginEntry, Registry


def manifest(name: str, kind: PluginKind = PluginKind.SOURCE, **kwargs) -> PluginManifest:
    return PluginManifest(name=name, kind=kind, **kwargs)


def entry(name: str, kind: PluginKind = PluginKind.SOURCE, **kwargs) -> PluginEntry:
    return PluginEntry(
        manifest=manifest(name, kind, **kwargs),
        build=lambda ctx, **params: {"name": name, "params": params},
        module=f"tests.fake.{name}",
    )


@pytest.fixture
def kernel() -> Kernel:
    return Kernel(Settings())


# ------------------------------------------------------------------ contracts


@pytest.mark.parametrize(
    ("handle", "expected"),
    [
        ("source:upstox", ("source", "upstox")),
        ("forecast:projection", ("forecast", "projection")),
    ],
)
def test_parse_handle_splits_once(handle, expected) -> None:
    assert parse_handle(handle) == expected


@pytest.mark.parametrize("handle", ["upstox", ":upstox", "source:", ""])
def test_parse_handle_rejects_malformed(handle) -> None:
    """A bare name is refused rather than guessed at across kinds."""
    with pytest.raises(ValueError):
        parse_handle(handle)


def test_manifest_accepts_a_mapping() -> None:
    """A plugin may declare its manifest as a dict, which keeps plugin.py declarative."""
    parsed = PluginManifest.from_mapping(
        {"name": "simulated", "kind": "source", "provides": ["bars"]}
    )
    assert parsed.kind is PluginKind.SOURCE
    assert parsed.provides == ("bars",)
    assert parsed.handle == "source:simulated"


def test_manifest_handle_carries_its_kind() -> None:
    """Handles are the addressing scheme, so the kind has to be in them."""
    assert manifest("simulated", PluginKind.SOURCE).handle == "source:simulated"
    assert manifest("projection", PluginKind.FORECAST).handle == "forecast:projection"


def test_manifest_requirements_default_to_empty() -> None:
    """A leaf plugin declares nothing and is still valid."""
    leaf = manifest("leaf")
    assert leaf.provides == ()
    assert leaf.requires == ()
    assert leaf.params == {}


# ------------------------------------------------------------------- registry


def test_registry_indexes_capabilities() -> None:
    registry = Registry()
    registry.register(entry("bars", provides=("bars",)))
    registry.register(entry("ticks", provides=("ticks",), kind=PluginKind.SOURCE))
    assert registry.provider("bars") == "source:bars"
    assert registry.provider("ticks") == "source:ticks"


def test_duplicate_handle_is_rejected() -> None:
    registry = Registry()
    registry.register(entry("same"))
    with pytest.raises(DuplicatePlugin, match="already registered"):
        registry.register(entry("same"))


def test_two_providers_of_one_capability_are_rejected_at_registration() -> None:
    """Ambiguity is a wiring bug, so it fails at registration rather than at use."""
    registry = Registry()
    registry.register(entry("one", provides=("bars",)))
    with pytest.raises(DuplicatePlugin, match="bars"):
        registry.register(entry("two", provides=("bars",)))


def test_capability_collision_across_kinds_is_rejected() -> None:
    """A capability resolving to two plugins would make wiring non-deterministic."""
    registry = Registry()
    registry.register(entry("a", provides=("bars",)))
    with pytest.raises(DuplicatePlugin, match="bars"):
        registry.register(entry("b", kind=PluginKind.TOOL, provides=("bars",)))


def test_unknown_handle_suggests_a_match() -> None:
    registry = Registry()
    registry.register(entry("upstox"))
    with pytest.raises(UnknownPlugin, match="source:upstox"):
        registry.get("source:upstox_live")


def test_bare_name_error_names_the_kind_prefix() -> None:
    registry = Registry()
    registry.register(entry("upstox"))
    with pytest.raises(UnknownPlugin, match="kind prefix"):
        registry.get("upstox")


def test_find_filters_by_kind_and_tag() -> None:
    registry = Registry()
    registry.register(entry("one", tags=("live",)))
    registry.register(entry("two", kind=PluginKind.RENDERER))
    assert [e.name for e in registry.find(PluginKind.SOURCE)] == ["one"]
    assert [e.name for e in registry.find(tag="live")] == ["one"]
    assert [e.name for e in registry.find(PluginKind.SOURCE, tag="offline")] == []


def test_consumers_of_lists_requirers() -> None:
    registry = Registry()
    registry.register(entry("provider", provides=("features",)))
    registry.register(entry("consumer", kind=PluginKind.FORECAST, requires=("features",)))
    assert registry.consumers_of("features") == ["forecast:consumer"]


def test_no_provider_error_lists_known_capabilities() -> None:
    registry = Registry()
    registry.register(entry("one", provides=("bars",)))
    with pytest.raises(NoProvider, match="bars"):
        registry.provider("conviction")


# ------------------------------------------------------------------------ bus


def test_bus_delivers_to_every_subscriber() -> None:
    bus = EventBus()
    seen: list[int] = []
    bus.subscribe(Topic.TICK, lambda payload: seen.append(payload))
    bus.subscribe("tick", lambda payload: seen.append(payload * 10))
    assert bus.publish(Topic.TICK, 2) == 2
    assert seen == [2, 20]


def test_bus_survives_a_failing_handler() -> None:
    """A handler that raises must not stop the feed or its peers."""
    bus = EventBus()
    seen: list[str] = []

    def broken(payload):
        raise ValueError("boom")

    bus.subscribe(Topic.TICK, broken, owner="broken")
    bus.subscribe(Topic.TICK, lambda payload: seen.append(payload))
    assert bus.publish(Topic.TICK, "tick-1") == 1
    assert seen == ["tick-1"]
    assert bus.failures == {"broken": 1}


def test_bus_counts_every_failure_without_flooding() -> None:
    """Every failure is counted, but only the first is logged.

    A handler that fails per tick would otherwise fill the log at tick rate.
    """
    bus = EventBus()
    bus.subscribe(Topic.TICK, lambda payload: 1 / 0, owner="divzero")
    for _ in range(5):
        bus.publish(Topic.TICK, 1)
    assert bus.failures == {"divzero": 5}
    assert bus.listeners(Topic.TICK) == 1


def test_cancelled_subscription_receives_nothing() -> None:
    bus = EventBus()
    seen: list[int] = []
    subscription = bus.subscribe(Topic.BAR, lambda payload: seen.append(payload))
    bus.publish(Topic.BAR, 1)
    subscription.cancel()
    bus.publish(Topic.BAR, 2)
    assert seen == [1]


def test_bus_counts_listeners_across_topics() -> None:
    bus = EventBus()
    bus.subscribe(Topic.TICK, lambda payload: None)
    bus.subscribe(Topic.SNAPSHOT, lambda payload: None)
    assert bus.listeners() == 2
    assert bus.listeners(Topic.TICK) == 1
    bus.clear()
    assert bus.listeners() == 0


# --------------------------------------------------------------------- kernel


def test_kernel_builds_and_memoises(kernel: Kernel) -> None:
    kernel.register(entry("simulated", provides=("bars",)))
    first = kernel.build("source:simulated")
    assert kernel.build("source:simulated") is first


def test_kernel_builds_fresh_when_parameterised(kernel: Kernel) -> None:
    """Explicit params mean "give me a new one", which is what tests and replays want."""
    kernel.register(entry("simulated"))
    first = kernel.build("source:simulated")
    second = kernel.build("source:simulated", days=5)
    assert first is not second
    assert second["params"] == {"days": 5}


def test_kernel_merges_configured_params() -> None:
    settings = Settings(plugin_config={"forecast:projection": {"bars_ahead": 3}})
    kernel = Kernel(settings)
    kernel.register(
        PluginEntry(
            manifest=manifest("projection", PluginKind.FORECAST),
            build=lambda ctx, **params: params,
            module="tests.fake.projection",
        )
    )
    assert kernel.build("forecast:projection") == {"bars_ahead": 3}


def test_capability_resolves_to_its_provider(kernel: Kernel) -> None:
    kernel.register(entry("bars", provides=("bars",)))
    assert kernel.capability("bars")["name"] == "bars"


def test_validate_reports_every_missing_requirement_at_once(kernel: Kernel) -> None:
    """A missing capability is a config mistake; show all of its consequences."""
    kernel.register(entry("a", kind=PluginKind.FORECAST, requires=("features", "conviction")))
    problems = kernel.validate()
    assert len(problems) == 2
    assert all("requires" in problem for problem in problems)


def test_validate_passes_once_the_provider_exists(kernel: Kernel) -> None:
    kernel.register(entry("features", kind=PluginKind.FEATURES, provides=("features",)))
    kernel.register(entry("consumer", kind=PluginKind.FORECAST, requires=("features",)))
    assert kernel.validate() == []


def test_require_raises_on_an_incomplete_composition(kernel: Kernel) -> None:
    kernel.register(entry("consumer", kind=PluginKind.FORECAST, requires=("features",)))
    with pytest.raises(MissingDependency):
        kernel.require(["forecast:consumer"])


def test_validate_treats_an_unknown_handle_as_a_problem(kernel: Kernel) -> None:
    assert "unknown plugin" in kernel.validate(["forecast:nope"])[0]


def test_teardown_stops_built_plugins() -> None:
    stopped: list[str] = []

    class Stoppable:
        def stop(self) -> None:
            stopped.append("stopped")

    kernel = Kernel(Settings())
    kernel.register(
        PluginEntry(
            manifest=manifest("stoppable", PluginKind.SOURCE),
            build=lambda ctx, **params: Stoppable(),
            module="tests.fake.stoppable",
        )
    )
    kernel.build("source:stoppable")
    kernel.teardown()
    assert stopped == ["stopped"]
    assert kernel.built() == {}


def test_build_failure_names_the_plugin() -> None:
    def explode(ctx, **params):
        raise RuntimeError("no token configured")

    kernel = Kernel(Settings())
    kernel.register(
        PluginEntry(
            manifest=manifest("bad", PluginKind.SOURCE),
            build=explode,
            module="tests.fake.bad",
        )
    )
    with pytest.raises(RuntimeError, match="source:bad"):
        kernel.build("source:bad")


# ----------------------------------------------------------- context plumbing


def test_context_exposes_settings_and_sibling_builds() -> None:
    kernel = Kernel(Settings())
    kernel.register(entry("bars", provides=("bars",)))
    kernel.register(
        PluginEntry(
            manifest=manifest("consumer", PluginKind.FORECAST, requires=("bars",)),
            build=lambda ctx, **params: {
                "symbol": ctx.settings.symbol,
                "bars": ctx.capability("bars")["name"],
                "provider": ctx.handle_for("bars"),
                "has": ctx.has_capability("bars"),
            },
            module="tests.fake.consumer",
        )
    )
    built = kernel.build("forecast:consumer")
    assert built == {
        "symbol": "NIFTY 50",
        "bars": "bars",
        "provider": "source:bars",
        "has": True,
    }


def test_context_is_what_a_builder_receives(kernel: Kernel) -> None:
    captured: dict = {}

    def build(ctx: PluginContext, **params):
        captured["type"] = type(ctx).__name__
        captured["params"] = ctx.params
        return object()

    kernel.register(
        PluginEntry(
            manifest=manifest("capture", PluginKind.TOOL),
            build=build,
            module="tests.fake.capture",
        )
    )
    kernel.build("tool:capture")
    assert captured == {"type": "PluginContext", "params": {}}


# ------------------------------------------------------------------- discovery


NESTED_PLUGIN = '''
from niftypulse.kernel import PluginManifest


MANIFEST = PluginManifest(
    name="{name}",
    kind="{kind}",
    provides={provides},
    requires={requires},
    description="nested test plugin",
)


def build(ctx, **params):
    return {{"name": "{name}", "params": params}}
'''


def write_plugin(root: Path, *parts: str, name: str, kind: str = "source",
                 provides: str = "", requires: str = "") -> Path:
    """Create a nested plugin package, with __init__.py at every level."""
    folder = root.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    for parent in [root, *[root.joinpath(*parts[: i + 1]) for i in range(len(parts))]]:
        (parent / "__init__.py").touch()
    (folder / f"{PLUGIN_MODULE}.py").write_text(
        textwrap.dedent(NESTED_PLUGIN.format(
            name=name,
            kind=kind,
            provides=_tuple_literal(provides),
            requires=_tuple_literal(requires),
        ))
    )
    return folder


def _tuple_literal(capability: str) -> str:
    return f'("{capability}",)' if capability else "()"


@pytest.fixture
def plugin_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A throwaway plugin package on sys.path, nested three levels deep."""
    package = tmp_path / "treekernel"
    package.mkdir()
    write_plugin(package, "sources", "nested", "deep", name="deep_source", provides="bars")
    write_plugin(package, "forecasts", "consumer", name="taker", kind="forecast",
                 requires="bars")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    importlib.invalidate_caches()
    return "treekernel"


def test_discovery_walks_arbitrarily_nested_packages(plugin_tree: str) -> None:
    """Nesting is the point: a plugin can be a folder of modules, at any depth."""
    from niftypulse.kernel.loader import discover

    report = discover(plugin_tree)
    assert report.ok, report.errors
    handles = sorted(entry.handle for entry in report.entries)
    assert handles == ["forecast:taker", "source:deep_source"]


def test_third_party_looking_pack_composes_against_capabilities(plugin_tree: str) -> None:
    """A discovered pack wires up on declared capabilities alone."""
    from niftypulse.kernel.loader import discover

    kernel = Kernel(Settings())
    for found in discover(plugin_tree).entries:
        kernel.register(found)

    assert kernel.validate() == []
    assert kernel.capability("bars")["name"] == "deep_source"


def test_load_module_rejects_a_module_without_a_manifest(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "nomanifest"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / f"{PLUGIN_MODULE}.py").write_text("def build(ctx, **params):\n    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    importlib.invalidate_caches()
    with pytest.raises(PluginLoadError, match="MANIFEST"):
        load_module("nomanifest.plugin")


def test_load_module_rejects_a_module_without_a_builder(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "nobuild"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / f"{PLUGIN_MODULE}.py").write_text(
        "from niftypulse.kernel import PluginManifest\n"
        "MANIFEST = PluginManifest(name='x', kind='source')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    importlib.invalidate_caches()
    with pytest.raises(PluginLoadError, match="build"):
        load_module("nobuild.plugin")


def test_broken_plugin_is_reported_not_raised(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad third-party pack must not stop the platform loading its own."""
    from niftypulse.kernel.loader import discover

    package = tmp_path / "brokenpack"
    package.mkdir()
    write_plugin(package, "good", name="good", provides="bars")
    write_plugin(package, "bad", name="bad")
    (package / "bad" / f"{PLUGIN_MODULE}.py").write_text("import definitely_not_a_module\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    importlib.invalidate_caches()

    report = discover("brokenpack")
    assert [entry.name for entry in report.entries] == ["good"]
    assert len(report.errors) == 1
    assert "definitely_not_a_module" in report.errors[0][1]


def test_entry_from_module_builds_a_registry_entry(plugin_tree: str) -> None:
    module = load_module(f"{plugin_tree}.sources.nested.deep.plugin")
    assert module.origin == "builtin"
    assert module.manifest.provides == ("bars",)
    assert module.module.endswith("plugin")


def test_bundled_tree_loads_cleanly() -> None:
    """Whatever ships in plugins/ must at least be importable and unambiguous.

    This is the guard that keeps a renamed or duplicated capability from landing
    in a release: the registry rejects a capability with two providers, so this
    test fails the moment two packs claim the same one.
    """
    from niftypulse.kernel.loader import BUILTIN_PACKAGE, discover

    report = discover(BUILTIN_PACKAGE)
    assert report.ok, report.errors
    registry = Registry()
    for found in report.entries:
        registry.register(found)


def test_bundled_tree_declares_only_known_kinds() -> None:
    from niftypulse.kernel.loader import BUILTIN_PACKAGE, discover

    for found in discover(BUILTIN_PACKAGE).entries:
        assert isinstance(found.manifest.kind, PluginKind)
        assert found.handle.startswith(found.manifest.kind.value)


def test_bootstrap_registers_discovered_plugins() -> None:
    kernel = Kernel.bootstrap(Settings(), with_entry_points=False)
    assert kernel.load_report.ok
    assert len(kernel.registry) == len(kernel.load_report.entries)
