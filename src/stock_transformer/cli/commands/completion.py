"""``stx completion`` — emit shell tab-completion scripts (bash, zsh, fish).

Click generates scripts that invoke ``stx`` with ``_STX_COMPLETE`` set; operators
install by sourcing the printed script from their shell profile so subcommands
and flags complete without maintaining hand-written completion files.
"""

from __future__ import annotations

import click
from click.shell_completion import BashComplete, FishComplete, ShellComplete, ZshComplete


def register_completion(root: click.Group) -> None:
    """Attach ``completion`` to the root ``stx`` group."""

    @root.command("completion")
    @click.argument(
        "shell",
        type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    )
    def completion_cmd(shell: str) -> None:
        """Print a shell script that enables tab completion for ``stx``.

        Pipe to a file and source it from ``~/.bashrc``, ``~/.zshrc``, or fish
        ``config.fish``. The script expects ``stx`` on ``PATH`` (same as the
        ``_STX_COMPLETE=…`` workflow in the README).
        """
        shell_key = shell.lower()
        # Import here so this module can load while ``app`` still registers subcommands.
        from stock_transformer.cli.app import cli as stx_cli

        cls: type[ShellComplete] = {
            "bash": BashComplete,
            "zsh": ZshComplete,
            "fish": FishComplete,
        }[shell_key]
        instance = cls(stx_cli, {}, prog_name="stx", complete_var="_STX_COMPLETE")
        click.echo(instance.source().rstrip())
