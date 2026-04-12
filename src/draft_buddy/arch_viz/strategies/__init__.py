"""
Pluggable extraction strategies for architecture diagrams.
"""

from draft_buddy.arch_viz.strategies.base import ExtractionStrategy
from draft_buddy.arch_viz.strategies.class_strategy import ClassStrategy
from draft_buddy.arch_viz.strategies.function_strategy import FunctionStrategy
from draft_buddy.arch_viz.strategies.module_strategy import ModuleStrategy

__all__ = [
    "ExtractionStrategy",
    "ModuleStrategy",
    "ClassStrategy",
    "FunctionStrategy",
]
