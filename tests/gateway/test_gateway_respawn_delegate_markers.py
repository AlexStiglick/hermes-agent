"""Gateway respawn boundaries must strip inherited delegate-child markers.

A gateway is never a delegate child; a marker inherited from the fenced shell
that (re)started it fences the gateway's embedded dispatcher. See #136152.
"""
from __future__ import annotations

import inspect
import os

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER


class TestStartupScrubHelper:
    def test_strips_both_markers_and_warns(self, monkeypatch, caplog):
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
        with caplog.at_level("WARNING", logger="hermes_cli.gateway_restart_env"):
            assert scrub_delegate_child_env_markers(os.environ) is True
        assert DELEGATED_CHILD_ENV_MARKER not in os.environ
        assert "HERMES_KANBAN_TASK" not in os.environ
        assert any("delegate child" in r.getMessage() for r in caplog.records)

    def test_clean_environment_is_a_silent_noop(self, monkeypatch, caplog):
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        with caplog.at_level("WARNING", logger="hermes_cli.gateway_restart_env"):
            assert scrub_delegate_child_env_markers(os.environ) is False
        assert not caplog.records

    def test_empty_marker_value_is_left_alone(self, monkeypatch):
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "")
        assert scrub_delegate_child_env_markers(os.environ) is False
        assert os.environ.get(DELEGATED_CHILD_ENV_MARKER) == ""


class TestWindowsDetachedSpawn:
    def test_spawn_detached_child_env_strips_delegate_markers(self, monkeypatch, tmp_path):
        from hermes_cli import gateway_windows as gw

        captured = {}

        class _FakeProc:
            pid = 4242

        def _fake_popen(argv, **kwargs):
            captured["env"] = kwargs["env"]
            return _FakeProc()

        monkeypatch.setattr(gw, "_assert_windows", lambda: None)
        monkeypatch.setattr(
            gw, "_build_gateway_argv", lambda home=None: (["hermes"], str(tmp_path), {})
        )
        monkeypatch.setattr(gw, "windows_detach_flags", lambda: 0)
        monkeypatch.setattr(gw, "windows_detach_flags_without_breakaway", lambda: 0)
        monkeypatch.setattr(gw, "_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(
            "tools.environments.local.served_profile_child_env",
            lambda **kwargs: dict(os.environ),
        )
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t")
        monkeypatch.setattr(gw.subprocess, "Popen", _fake_popen)

        assert gw._spawn_detached() == 4242
        assert DELEGATED_CHILD_ENV_MARKER not in captured["env"]
        assert "HERMES_KANBAN_TASK" not in captured["env"]


class TestPosixWatcherSource:
    def test_watcher_pops_inherited_markers_before_respawn(self):
        """Static contract: watcher source pops markers before respawn."""
        from hermes_cli import gateway as gw

        src = inspect.getsource(gw._spawn_gateway_restart_watcher)
        assert "os.environ.pop" in src
        assert "DELEGATED_CHILD_ENV_MARKER" in src


class TestParentTerminalScrubUsesContextVar:
    def test_stale_env_marker_does_not_stamp_via_scrub(self, monkeypatch):
        """Parent with stale env marker must not call scrub_kanban_env."""
        from tools.environments.local import _scrub_delegated_child_kanban_env

        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        env = {
            DELEGATED_CHILD_ENV_MARKER: "1",
            "HERMES_KANBAN_TASK": "t_parent",
            "PATH": "/bin",
        }
        out = _scrub_delegated_child_kanban_env(dict(env))
        assert out.get("HERMES_KANBAN_TASK") == "t_parent"
        assert out.get(DELEGATED_CHILD_ENV_MARKER) == "1"

    def test_live_contextvar_scrubs_and_stamps(self, monkeypatch):
        from agent.delegation_context import delegated_child_context
        from tools.environments.local import _scrub_delegated_child_kanban_env

        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
        env = {
            "HERMES_KANBAN_TASK": "t_parent",
            "HERMES_KANBAN_RUN_ID": "1",
            "PATH": "/bin",
        }
        with delegated_child_context():
            out = _scrub_delegated_child_kanban_env(dict(env))
        assert out.get(DELEGATED_CHILD_ENV_MARKER)
        assert "HERMES_KANBAN_TASK" not in out


class TestCronMustNotSilentlyUnfence:
    """G (gateway module): cron must not reuse the gateway scrub helper."""

    def test_cron_scheduler_does_not_import_or_call_gateway_scrub(self):
        import inspect as _inspect

        import cron.scheduler as scheduler

        src = _inspect.getsource(scheduler)
        assert "scrub_delegate_child_env_markers" not in src
        assert "gateway_restart_env" not in src
