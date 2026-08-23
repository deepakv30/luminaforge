"""Smoke tests for the adaptive UI shell. Does not launch a server."""

from __future__ import annotations

from pathlib import Path

from ui.layout import NAV_GROUPS, TAB_IDS
from ui.theme import CSS_DIR, get_css_paths, get_head_html, get_runtime_css, get_theme


def test_should_open_browser_skips_when_no_browser(monkeypatch):
    from app import _should_open_browser

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("LUMINAFORGE_INBROWSER", raising=False)
    monkeypatch.setattr("app.shutil.which", lambda name: None)
    assert _should_open_browser() is False
    monkeypatch.setenv("LUMINAFORGE_INBROWSER", "1")
    assert _should_open_browser() is True


def test_audio_payload_normalizes_to_path():
    from ui.studios.voice import _audio_to_path, _safe_profile_name

    assert _audio_to_path(None) is None
    assert _audio_to_path("") is None
    assert _audio_to_path("/tmp/ref.wav") == "/tmp/ref.wav"
    assert _audio_to_path({"path": "/tmp/clip.mp3", "orig_name": "clip.mp3"}) == "/tmp/clip.mp3"
    assert _safe_profile_name("  My Narrator!  ", None) == "My_Narrator"
    assert _safe_profile_name("", "/tmp/deep_voice.wav") == "deep_voice"


def test_mode_badge_html_includes_xai():
    from ui.layout import mode_badge_html

    assert "status-xai" in mode_badge_html("xai")
    assert "status-cloud" in mode_badge_html("ollama_cloud")
    assert "status-local" in mode_badge_html("ollama")


def test_nav_ids_cover_all_studios():
    assert "dashboard" in TAB_IDS
    assert "settings" in TAB_IDS
    assert len(TAB_IDS) == 10
    grouped = [tid for _, items in NAV_GROUPS for tid, _ in items]
    assert grouped == TAB_IDS


def test_css_files_exist():
    paths = get_css_paths()
    assert len(paths) == 3
    for p in paths:
        assert p.exists(), p
        text = p.read_text(encoding="utf-8")
        assert len(text) > 50
    assert (CSS_DIR / "tokens.css").exists()


def test_css_has_breakpoints_and_owned_classes():
    blob = "\n".join(p.read_text(encoding="utf-8") for p in get_css_paths())
    assert "@media (max-width: 639px)" in blob
    assert "@media (min-width: 1024px)" in blob
    assert "@media (min-width: 1921px)" in blob
    assert ".studio-shell" in blob
    assert ".gallery-fluid" in blob
    assert "#lf-main-tabs" in blob
    assert "safe-area-inset" in blob
    assert "hover: hover" in blob


def test_runtime_css_uses_config_keys():
    css = get_runtime_css({"ui": {"content_max_width": 1400, "gallery_min_tile": 180, "gallery_columns": 4}})
    assert "1400px" in css
    assert "180px" in css
    assert "--lf-gallery-max-cols: 4" in css


def test_head_has_mobile_viewport():
    head = get_head_html()
    assert "viewport-fit=cover" in head
    assert "theme-color" in head


def test_theme_builds():
    theme = get_theme()
    assert theme is not None


def test_log_op_times_success_and_failure(capsys):
    from ui.services import log_op

    with log_op("generate text", model="demo"):
        pass
    out = capsys.readouterr().out
    assert "[LuminaForge] generate text ok" in out
    assert "model=demo" in out

    try:
        with log_op("generate image"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    err_out = capsys.readouterr().out
    assert "[LuminaForge] generate image failed" in err_out
    assert "RuntimeError: boom" in err_out


def test_nav_and_search_skip_the_generation_queue():
    from app import create_app

    demo = create_app()
    by_name = {}
    for fn in demo.fns.values():
        name = getattr(fn, "api_name", None) or getattr(getattr(fn, "fn", None), "__name__", "")
        by_name.setdefault(name, []).append(bool(fn.queue))

    assert by_name.get("do_generate") == [True]
    assert by_name.get("generate_images_with_progress") == [True]
    assert by_name.get("pull_model") == [True]
    assert by_name.get("load_recent_gallery_items") == [False]

    search_flags = [q for name, flags in by_name.items() if name.startswith("go_search") for q in flags]
    assert search_flags and all(q is False for q in search_flags)

    select_flags = [q for name, flags in by_name.items() if name.startswith("_select") for q in flags]
    assert len(select_flags) == 10
    assert all(q is False for q in select_flags)


def test_create_app_registers_shell():
    from app import create_app

    demo = create_app()
    blocks = getattr(demo, "blocks", {}) or {}
    elem_ids = set()
    classes = set()
    tab_ids = set()
    for block in blocks.values():
        eid = getattr(block, "elem_id", None)
        if eid:
            elem_ids.add(eid)
        ecs = getattr(block, "elem_classes", None) or []
        if isinstance(ecs, str):
            ecs = [ecs]
        classes.update(ecs)
        tid = getattr(block, "id", None)
        if isinstance(tid, str):
            tab_ids.add(tid)

    assert "lf-main-tabs" in elem_ids
    assert "lf-sidebar" in elem_ids
    assert "nav-dashboard" in elem_ids
    assert "nav-settings" in elem_ids
    assert "main-tabs" in classes
    for expected in TAB_IDS:
        assert expected in tab_ids, f"missing tab id {expected}"
