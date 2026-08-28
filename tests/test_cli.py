from microserve import cli
from microserve.cli import CommandSpec


def test_top_level_help_lists_the_unified_commands(capsys) -> None:
    cli.main([])

    output = capsys.readouterr().out
    assert "quickstart" in output
    assert "generate" in output
    assert "bench" in output
    assert "scorecard" in output
    assert "fetch" in output


def test_dispatch_preserves_subcommand_arguments_and_program_name(monkeypatch) -> None:
    called = {}

    def fake_main(arguments, *, prog):
        called["arguments"] = arguments
        called["prog"] = prog

    monkeypatch.setitem(cli.COMMANDS, "bench", CommandSpec("test", fake_main))

    cli.main(["bench", "--batch-size", "8"])

    assert called == {
        "arguments": ["--batch-size", "8"],
        "prog": "microserve bench",
    }
