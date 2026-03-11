"""Command subpackage for vector embedding CLI."""

from .degrade import register_degrade_command
from .generate import register_generate_command
from .load import register_load_command
from .plot import register_plot_command
from .query import register_query_command
from .reindex import register_reindex_command

__all__ = [
    "register_degrade_command",
    "register_generate_command",
    "register_load_command",
    "register_plot_command",
    "register_query_command",
    "register_reindex_command",
]
