"""Annotated semantic map (ASM) side-channel for OpenFrontier.

A top-down semantic map is accumulated continuously from the observations the
navigation loop is already taking, annotated with object-name labels in the
style of MapNav (ACL 2025), and written to disk. Nothing in the navigation
pipeline is modified or consulted -- the map is a pure by-product.

Typical use, with no edits to any existing file:

    OF_ASM=1 python -m asm.run_with_asm benchmark --config config/navigation.yaml ...

Or programmatically:

    from asm import ASMConfig, attach
    builder = attach(agent, ASMConfig(resolution_m=0.05))
    image, objects, sentence = builder.latest()
"""
from .builder import ASMBuilder
from .config import DEFAULT_CATEGORIES, ASMConfig, asm_enabled
from .hook import attach, detach
from .semantic_map import AnnotatedSemanticMap, LabeledObject

__all__ = [
    "ASMBuilder",
    "ASMConfig",
    "AnnotatedSemanticMap",
    "LabeledObject",
    "DEFAULT_CATEGORIES",
    "asm_enabled",
    "attach",
    "detach",
]
