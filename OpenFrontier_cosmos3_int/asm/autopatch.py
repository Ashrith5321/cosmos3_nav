"""Import this module to give every NavigationAgent an ASM, with zero edits.

    OF_ASM=1 python -m asm.run_with_asm benchmark --config config/navigation.yaml ...

or, equivalently, from any driver script:

    import asm.autopatch   # noqa: F401  (must precede agent construction)

Patching is a no-op unless OF_ASM is truthy, so importing it is always safe.
"""
import os

from .config import ASMConfig, asm_enabled
from .hook import attach

_PATCHED = False


def install(cfg: ASMConfig = None, force: bool = False) -> bool:
    """Wrap NavigationAgent.__init__ so each new agent gets a builder."""
    global _PATCHED
    if _PATCHED:
        return True
    if not force and not asm_enabled():
        return False

    try:
        from nav.agent import NavigationAgent
    except Exception as exc:  # pragma: no cover - import path problem
        print(f"ASM: cannot patch NavigationAgent ({exc!r}); ASM disabled")
        return False

    original_init = NavigationAgent.__init__
    config = cfg or ASMConfig.from_env()

    def init_with_asm(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            attach(self, config)
        except Exception as exc:
            print(f"ASM: attach raised, continuing without it: {exc!r}")

    init_with_asm.__wrapped__ = original_init
    NavigationAgent.__init__ = init_with_asm
    _PATCHED = True
    print(f"ASM: enabled (categories={config.categories}, "
          f"res={config.resolution_m} m, segment_every={config.segment_every_n_frames})")
    return True


install()
