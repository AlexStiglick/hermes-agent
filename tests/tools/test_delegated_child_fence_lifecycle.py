"""Regression: delegated-child fence lifecycle (false-positive + fail-closed).

Covers the fix that:
* keeps independent CLI roots unfenced (no sticky child identity)
* fences real delegate_task children and ordinary descendants that inherit
  ``HERMES_DELEGATED_CHILD_CONTEXT`` (fail-closed; no silent unfence)
* restores parent ContextVar identity after child completion / exceptions
* excludes scope markers from terminal session snapshots
* scrubs markers at gateway spawn/startup only (not cron)
* lets read-only kanban CLI verbs succeed under a fence when the DB exists

Scenarios A–J are named in the test docstrings for the PR mapping.
"""
from __future__ import annotations

import argparse
import inspect
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_context,
    delegated_child_subprocess_env,
    is_delegated_child_context,
    is_delegated_child_process_context,
    scrub_kanban_env,
)
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kbc._INITIALIZED_PATHS.clear()
    kbc.init_db()
    return home


def _parse_kanban(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser.parse_args(["kanban", *argv])


def _seed_task(title: str = "seed") -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title=title, assignee="root-worker")


def _fresh_process_probe(*, env: dict[str, str], code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a brand-new interpreter with *env* (plus PYTHONPATH)."""
    full_env = dict(env)
    full_env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), full_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    full_env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(
        [sys.executable, "-c", code],
        env=full_env,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=30,
        check=False,
    )


def _marker_is_set(env: dict) -> bool:
    return bool(env.get(DELEGATED_CHILD_ENV_MARKER))


class TestIndependentCliRoot:
    """A: no child marker → Kanban reads + mutations use non-child permissions."""

    def test_a_root_reads_and_mutations_unfenced(self, kanban_home, monkeypatch):
        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
        assert not is_delegated_child_context()
        assert not is_delegated_child_process_context()

        tid = _seed_task("root-task")
        assert kc.kanban_command(_parse_kanban("list")) == 0
        assert kc.kanban_command(_parse_kanban("show", tid)) == 0
        assert kc.kanban_command(_parse_kanban("stats")) == 0

        assert kc.kanban_command(
            _parse_kanban("create", "root-created", "--assignee", "root-worker")
        ) == 0
        kb.create_board("root-board")
        assert kb.board_exists("root-board")


class TestDelegatedChildRestrictions:
    """B: unauthorized mutation rejected; ordinary subprocess cannot bypass."""

    def test_b_contextvar_blocks_db_and_cli_mutation(self, kanban_home):
        _seed_task()
        with delegated_child_context():
            assert is_delegated_child_context()
            with kbc.connect_closing() as conn:
                with pytest.raises(PermissionError, match="delegate_task child"):
                    kb.create_task(conn, title="child-leak")
            rc = kc.kanban_command(
                _parse_kanban("create", "child-cli", "--assignee", "x")
            )
            assert rc == 1

    def test_b_env_marker_alone_fences_without_contextvar(self, kanban_home, monkeypatch):
        """Fail-closed lineage: inherited env marker fences even sans ContextVar."""
        _seed_task()
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        assert not is_delegated_child_context()
        assert is_delegated_child_process_context()

        with kbc.connect_closing() as conn:
            with pytest.raises(PermissionError, match="delegate_task child"):
                kb.create_task(conn, title="env-only-leak")

        rc = kc.kanban_command(_parse_kanban("create", "env-cli", "--assignee", "x"))
        assert rc == 1

    def test_b_ordinary_subprocess_cannot_bypass_via_import(
        self, kanban_home, monkeypatch
    ):
        """A fresh process that inherits the marker cannot mutate by importing DB APIs."""
        home = str(kanban_home)
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        code = textwrap.dedent(
            f"""
            import os
            os.environ["HERMES_HOME"] = {home!r}
            os.environ[{DELEGATED_CHILD_ENV_MARKER!r}] = "1"
            from hermes_cli import kanban_db as kb
            from hermes_cli import kanban_db_connect as kbc
            kbc._INITIALIZED_PATHS.clear()
            try:
                with kbc.connect_closing() as conn:
                    kb.create_task(conn, title="bypass")
            except PermissionError as exc:
                print("DENIED:" + str(exc))
            else:
                print("ALLOWED")
            """
        )
        env = os.environ.copy()
        env[DELEGATED_CHILD_ENV_MARKER] = "1"
        env["HERMES_HOME"] = home
        proc = _fresh_process_probe(env=env, code=code)
        assert proc.returncode == 0, proc.stderr
        assert "DENIED:" in proc.stdout
        assert "ALLOWED" not in proc.stdout


class TestFenceIdentityLifecycle:
    def test_c_parent_restored_after_child_completes(self, kanban_home, monkeypatch):
        """C: parent context restoration after delegated child completes."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        assert not is_delegated_child_context()

        with delegated_child_context("child-session"):
            assert is_delegated_child_context()
            assert is_delegated_child_process_context()
            from tools.environments.local import build_subprocess_env

            child_env = build_subprocess_env(
                base={"HERMES_KANBAN_TASK": "t_parent", "PATH": os.environ.get("PATH", "")}
            )
            assert _marker_is_set(child_env)
            assert "HERMES_KANBAN_TASK" not in child_env

        assert not is_delegated_child_context()
        assert os.environ.get("HERMES_KANBAN_TASK") == "t_parent"
        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="after-child")
        assert tid

    def test_d_sequential_delegation_identity_correct(self):
        """D: sequential children each fence; gaps between them are parent."""
        observations: list[tuple[str, bool]] = []

        for label in ("child-a", "child-b", "child-c"):
            assert not is_delegated_child_context()
            with delegated_child_context(label):
                observations.append((label, is_delegated_child_context()))
            observations.append((f"after-{label}", is_delegated_child_context()))

        assert observations == [
            ("child-a", True),
            ("after-child-a", False),
            ("child-b", True),
            ("after-child-b", False),
            ("child-c", True),
            ("after-child-c", False),
        ]

    def test_e_exception_leaves_no_sticky_child_identity(self):
        """E: exceptions / cancellation leave no sticky child identity."""
        with pytest.raises(RuntimeError, match="boom"):
            with delegated_child_context("exploding-child"):
                assert is_delegated_child_context()
                raise RuntimeError("boom")

        assert not is_delegated_child_context()
        assert not is_delegated_child_process_context()

    def test_e_generator_close_cancels_without_sticky_identity(self):
        """E: generator close (cancellation-shaped) also restores the fence."""

        def _gen():
            with delegated_child_context("cancelled-child"):
                assert is_delegated_child_context()
                yield "inside"

        g = _gen()
        assert next(g) == "inside"
        g.close()
        assert not is_delegated_child_context()


class TestTerminalEnvSnapshots:
    """F: dump excludes markers; child sees marker; parent next command does not."""

    def test_f_export_dump_unsets_scope_markers(self):
        from tools.environments.base_session_env import _export_dump_excluding_session_vars

        snippet = _export_dump_excluding_session_vars('"$__hermes_snap_tmp"')
        assert "HERMES_DELEGATED_CHILD_CONTEXT" in snippet
        assert "HERMES_CRON_SESSION" in snippet
        assert "unset" in snippet
        assert "grep -vE" not in snippet

    def test_f_child_command_env_stamped_parent_next_not(
        self, monkeypatch, tmp_path
    ):
        from tools.environments.local import build_subprocess_env

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
        base = {
            "HERMES_KANBAN_TASK": "t_parent",
            "HERMES_KANBAN_RUN_ID": "9",
            "PATH": os.environ.get("PATH", ""),
            "SystemRoot": os.environ.get("SystemRoot", ""),
        }

        with delegated_child_context():
            child_env = build_subprocess_env(base=dict(base))
        assert _marker_is_set(child_env)
        assert "HERMES_KANBAN_TASK" not in child_env

        parent_env = build_subprocess_env(base=dict(base))
        assert DELEGATED_CHILD_ENV_MARKER not in parent_env
        assert parent_env.get("HERMES_KANBAN_TASK") == "t_parent"

        # Stale process env marker must not re-stamp via terminal scrub.
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        stale_parent = build_subprocess_env(
            base={**base, DELEGATED_CHILD_ENV_MARKER: "1"}
        )
        assert stale_parent.get("HERMES_KANBAN_TASK") == "t_parent"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
    def test_f_shared_snapshot_file_excludes_markers_after_child_command(
        self, tmp_path, monkeypatch
    ):
        from tools.environments.local import LocalEnvironment

        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
        env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
        env.init_session()
        try:
            with delegated_child_context():
                child_out = env.execute(
                    f'echo "CHILD=[{DELEGATED_CHILD_ENV_MARKER}=${DELEGATED_CHILD_ENV_MARKER}]"'
                )
            parent_out = env.execute(
                f'echo "PARENT=[{DELEGATED_CHILD_ENV_MARKER}=${DELEGATED_CHILD_ENV_MARKER}]"'
            )

            child_text = child_out.get("output", "")
            parent_text = parent_out.get("output", "")
            assert re.search(
                r"CHILD=\[HERMES_DELEGATED_CHILD_CONTEXT=.+\]", child_text
            ), child_text
            assert re.search(
                r"PARENT=\[HERMES_DELEGATED_CHILD_CONTEXT=\]", parent_text
            ), parent_text

            snap = env._snapshot_path
            if os.path.exists(snap):
                with open(snap, encoding="utf-8", errors="replace") as fh:
                    body = fh.read()
                assert "HERMES_DELEGATED_CHILD_CONTEXT" not in body
                assert "HERMES_CRON_SESSION" not in body
        finally:
            env.cleanup()


class TestCronGatewayLifecycle:
    """G: gateway scrub verified; cron must not silently unfence descendants."""

    def test_g_gateway_scrub_strips_markers(self, monkeypatch, caplog):
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
        with caplog.at_level("WARNING", logger="hermes_cli.gateway_restart_env"):
            assert scrub_delegate_child_env_markers(os.environ) is True
        assert DELEGATED_CHILD_ENV_MARKER not in os.environ
        assert "HERMES_KANBAN_TASK" not in os.environ

    def test_g_cron_modules_do_not_call_gateway_scrub(self):
        """Cron must not reuse the gateway spawn-edge unfence helper."""
        import cron.scheduler as scheduler

        src = inspect.getsource(scheduler)
        assert "scrub_delegate_child_env_markers" not in src
        assert "gateway_restart_env" not in src

    def test_g_inherited_marker_without_spawn_edge_grant_stays_fenced(
        self, monkeypatch
    ):
        """Fail-closed: env marker alone still fences (no silent cron unfence)."""
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        assert not is_delegated_child_context()
        assert is_delegated_child_process_context()

        from agent.delegation_context import non_dispatcher_owned_context

        with non_dispatcher_owned_context():
            assert is_delegated_child_process_context()
            assert os.environ.get(DELEGATED_CHILD_ENV_MARKER) == "1"


class TestKanbanReadOnlyUnderFence:
    """H: show/list/stats succeed under child fence when the DB already exists."""

    @pytest.mark.parametrize(
        "argv_factory",
        [
            lambda tid: ("list",),
            lambda tid: ("show", tid),
            lambda tid: ("stats",),
        ],
        ids=["list", "show", "stats"],
    )
    def test_h_readonly_verbs_under_contextvar(
        self, kanban_home, argv_factory, monkeypatch
    ):
        tid = _seed_task("readable")
        init_calls: list[str] = []
        real_init = kb.init_db

        def tracking_init(*args, **kwargs):
            init_calls.append("called")
            return real_init(*args, **kwargs)

        monkeypatch.setattr(kb, "init_db", tracking_init)
        monkeypatch.setattr(kc.kb, "init_db", tracking_init)
        monkeypatch.setattr(kbc, "init_db", tracking_init)

        with delegated_child_context():
            rc = kc.kanban_command(_parse_kanban(*argv_factory(tid)))
        assert rc == 0
        assert init_calls == []

    def test_h_readonly_verbs_under_env_marker_only(self, kanban_home, monkeypatch):
        tid = _seed_task("env-readable")
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        assert kc.kanban_command(_parse_kanban("list")) == 0
        assert kc.kanban_command(_parse_kanban("show", tid)) == 0
        assert kc.kanban_command(_parse_kanban("stats")) == 0

    def test_h_empty_backfill_skips_write_txn_under_fence(
        self, kanban_home, monkeypatch
    ):
        """Empty run-backfill must not enter write_txn (would trip the fence)."""
        from hermes_cli import kanban_db_connect as kbc

        _seed_task("no-backfill-needed")
        kbc._INITIALIZED_PATHS.clear()
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")

        real_write_txn = kbc.write_txn
        calls = {"n": 0}

        def counting_write_txn(conn, *, allow_nested=False):
            calls["n"] += 1
            return real_write_txn(conn, allow_nested=allow_nested)

        monkeypatch.setattr(kbc, "write_txn", counting_write_txn)
        monkeypatch.setattr(kb, "write_txn", counting_write_txn)
        conn = kbc.connect()
        try:
            rows = kb.list_tasks(conn, limit=10)
            assert any(r.title == "no-backfill-needed" for r in rows)
        finally:
            conn.close()
        assert calls["n"] == 0


class TestFreshProcessAuthority:
    """I: a new process fences only from its own env / ContextVar — not parent."""

    def test_i_fresh_process_without_marker_is_unfenced_despite_parent_context(
        self, kanban_home, monkeypatch
    ):
        home = str(kanban_home)
        code = textwrap.dedent(
            f"""
            import os
            os.environ["HERMES_HOME"] = {home!r}
            os.environ.pop({DELEGATED_CHILD_ENV_MARKER!r}, None)
            from agent.delegation_context import (
                is_delegated_child_context,
                is_delegated_child_process_context,
            )
            print(
                "ctx=" + str(is_delegated_child_context())
                + " proc=" + str(is_delegated_child_process_context())
            )
            """
        )
        with delegated_child_context():
            assert is_delegated_child_context()
            env = {
                k: v
                for k, v in os.environ.items()
                if k != DELEGATED_CHILD_ENV_MARKER
            }
            env["HERMES_HOME"] = home
            proc = _fresh_process_probe(env=env, code=code)
        assert proc.returncode == 0, proc.stderr
        assert "ctx=False proc=False" in proc.stdout

    def test_i_fresh_process_with_marker_fences_from_own_env(self, kanban_home):
        home = str(kanban_home)
        code = textwrap.dedent(
            f"""
            import os
            os.environ["HERMES_HOME"] = {home!r}
            os.environ[{DELEGATED_CHILD_ENV_MARKER!r}] = "1"
            from agent.delegation_context import (
                is_delegated_child_context,
                is_delegated_child_process_context,
            )
            print(
                "ctx=" + str(is_delegated_child_context())
                + " proc=" + str(is_delegated_child_process_context())
            )
            """
        )
        env = os.environ.copy()
        env[DELEGATED_CHILD_ENV_MARKER] = "1"
        env["HERMES_HOME"] = home
        proc = _fresh_process_probe(env=env, code=code)
        assert proc.returncode == 0, proc.stderr
        assert "ctx=False proc=True" in proc.stdout

    def test_i_spawn_edge_grant_propagates_marker(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
        with delegated_child_context():
            granted = delegated_child_subprocess_env({"HERMES_KANBAN_TASK": "t_parent"})
        assert granted is not None
        assert _marker_is_set(granted)
        assert "HERMES_KANBAN_TASK" not in granted


class TestWindowsProcessEnvPropagation:
    """J: Windows-practical checks for process creation env scrubbing."""

    def test_j_windows_detached_spawn_strips_markers(self, monkeypatch, tmp_path):
        from hermes_cli import gateway_windows as gw

        captured: dict = {}

        class _FakeProc:
            pid = 5150

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

        assert gw._spawn_detached() == 5150
        assert DELEGATED_CHILD_ENV_MARKER not in captured["env"]
        assert "HERMES_KANBAN_TASK" not in captured["env"]

    def test_j_subprocess_popen_receives_scrubbed_child_env(self, monkeypatch):
        """Practical CreateProcess-shaped check via subprocess + scrubbed env."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

        with delegated_child_context():
            child_env = scrub_kanban_env(dict(os.environ))
            if sys.platform == "win32":
                for key in ("SystemRoot", "SYSTEMROOT", "PATH", "Path"):
                    if key in os.environ and key not in child_env:
                        child_env[key] = os.environ[key]
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT','')); "
                    "print(os.environ.get('HERMES_KANBAN_TASK','MISSING'))",
                ],
                env=child_env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        assert proc.returncode == 0, proc.stderr
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        assert lines[0]  # path-valued or legacy "1"
        assert lines[1] == "MISSING"
