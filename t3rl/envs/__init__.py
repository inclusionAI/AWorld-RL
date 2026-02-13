"""Environment-specific helper utilities."""

from t3rl.envs.bfcl_gym import BFCLGymAdapter

__all__ = [
    "BFCLEnv",
    "BFCLGymAdapter",
]


def __getattr__(name: str):
    if name == "BFCLEnv":
        from t3rl.envs.bfcl import BFCLEnv

        return BFCLEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
