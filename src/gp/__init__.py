"""gp package."""

__version__ = "0.1.0"

__all__ = ["__version__", "greet"]


def greet(name: str = "world") -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"
