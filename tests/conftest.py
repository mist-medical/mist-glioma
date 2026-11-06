"""Shared test configuration."""

import sys
import types


class ANTsImage:
    """Type placeholder used while antspyx is unavailable in the test environment."""


ants_stub = types.ModuleType("ants")
ants_stub.core = types.SimpleNamespace(
    ants_image=types.SimpleNamespace(ANTsImage=ANTsImage)
)
sys.modules.setdefault("ants", ants_stub)

greedy_stub = types.ModuleType("picsl_greedy")
greedy_stub.Greedy3D = object
sys.modules.setdefault("picsl_greedy", greedy_stub)
