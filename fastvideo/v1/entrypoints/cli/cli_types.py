import argparse
from typing import Any


class CLISubcommand:
    """Base class for CLI subcommands"""
    
    def __init__(self):
        self.name = ""
    
    def cmd(self, args: argparse.Namespace) -> None:
        """Execute the command with the given arguments"""
        raise NotImplementedError
    
    def validate(self, args: argparse.Namespace) -> None:
        """Validate the arguments for this command"""
        pass
    
    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """Initialize the subparser for this command"""
        raise NotImplementedError 