# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/main.py
from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.entrypoints.cli.generate import cmd_init as generate_cmd_init
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.entrypoints.cli.serve import cmd_init as serve_cmd_init
from fastvideo.entrypoints.cli.bench import cmd_init as bench_cmd_init
from fastvideo.entrypoints.cli.eval import cmd_init as eval_cmd_init


def cmd_init() -> list[CLISubcommand]:
    """Initialize all commands from separate modules"""
    commands = []
    commands.extend(generate_cmd_init())
    commands.extend(serve_cmd_init())
    commands.extend(bench_cmd_init())
    commands.extend(eval_cmd_init())
    return commands


def main() -> None:
    parser = FlexibleArgumentParser(description="FastVideo CLI")
    parser.add_argument('-v', '--version', action='version', version='0.1.0')

    subparsers = parser.add_subparsers(required=False, dest="subparser")

    cmds = {}
    for cmd in cmd_init():
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        cmds[cmd.name] = cmd

    args, unknown = parser.parse_known_args()
    if unknown and args.subparser not in {"generate", "serve"}:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args._unknown = unknown
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)
        args.dispatch_function(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
