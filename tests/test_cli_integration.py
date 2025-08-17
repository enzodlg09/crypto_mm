from __future__ import annotations

from crypto_mm.main import build_parser


def test_help_and_subcommands_exist() -> None:
    parser = build_parser()
    subs = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    assert set(subs.keys()) == {"live", "replay", "plot", "check", "ws-ping", "ws-trades", "ws-book"}
