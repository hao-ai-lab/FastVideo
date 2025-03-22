import argparse
from typing import List

# Import commands
from fastvideo.v1.entrypoints.cli.serve import cmd_init as serve_cmd_init
from fastvideo.v1.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.v1.entrypoints.cli import utils


def cmd_init() -> List[CLISubcommand]:
    """Initialize all commands from separate modules"""
    commands = []
    commands.extend(serve_cmd_init())
    return commands


def main():
    # Create the main parser
    parser = argparse.ArgumentParser(description="FastVideo CLI")
    parser.add_argument('-v', '--version', action='version', version='0.1.0')
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(required=False, dest="command")
    
    # Register commands
    cmds = {}
    for cmd in cmd_init():
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        cmds[cmd.name] = cmd
    
    # Parse known arguments first
    args, unknown = parser.parse_known_args()
    
    # Store unknown arguments
    args.unknown_args = unknown
    
    # Validate command if it exists
    if args.command in cmds:
        cmds[args.command].validate(args)
    
    # Dispatch to the appropriate command
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 