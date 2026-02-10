#!/usr/bin/env uv run
"""
Vector embedding CLI tool with generate, load, and query functionality.
"""
import argparse

from commands import (
    register_generate_command,
    register_load_command,
    register_plot_command,
    register_query_command,
)


def main():
    parser = argparse.ArgumentParser(
        description='Vector embedding CLI tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # Register subcommands
    register_generate_command(subparsers)
    register_load_command(subparsers)
    register_plot_command(subparsers)
    register_query_command(subparsers)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
