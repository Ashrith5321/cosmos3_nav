"""Entry point that enables the ASM and then runs an existing driver unchanged.

    OF_ASM=1 python -m asm.run_with_asm benchmark --config config/navigation.yaml \\
        --output-path output --eval_episodes 28 --max_steps 500

Everything after the module name is forwarded verbatim as that module's argv, so
this is a drop-in replacement for `python benchmark.py ...` that leaves
benchmark.py untouched.
"""
import runpy
import sys
from pathlib import Path

# Allow `python asm/run_with_asm.py` as well as `python -m asm.run_with_asm`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)

    from asm.autopatch import install
    if not install(force=True):
        print("ASM: could not install; running the driver unmodified")

    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    if module.endswith(".py"):
        runpy.run_path(module, run_name="__main__")
    else:
        runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
