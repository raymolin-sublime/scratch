"""Command subpackage for vector embedding CLI."""
from .generate import register_generate_command
from .load import register_load_command
from .plot import register_plot_command
from .query import register_query_command

__all__ = [
    'register_generate_command',
    'register_load_command',
    'register_plot_command',
    'register_query_command',
]
