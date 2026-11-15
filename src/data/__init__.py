"""Data loading, synthetic instance generation, Toronto parsing, and validation (P2)."""

from pathlib import Path

from src.core.instance import Instance
from src.data.generator import (
    generate_all_from_yaml,
    generate_instance,
    load_instance_npz,
    save_instance_npz,
)
from src.data.stats import compute_instance_stats, summarize_instances
from src.data.toronto_parser import parse_crs_file, parse_toronto_instance
from src.data.validate import (
    compute_sha256,
    validate_instance,
    verify_checksums,
    write_checksums,
)


def load_instance(name: str, generated_dir: str | Path = "data/generated") -> Instance:
    """Load a generated benchmark instance by name (e.g., 'small-a', 'medium-b')."""
    # Build the full path to the compressed .npz file
    path = Path(generated_dir) / f"{name}.npz"
    # Raise a clear error if the requested instance file does not exist on disk
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")
    # Delegate reading and reconstruction to load_instance_npz
    return load_instance_npz(path)


__all__ = [
    "compute_instance_stats",
    "compute_sha256",
    "generate_all_from_yaml",
    "generate_instance",
    "load_instance",
    "load_instance_npz",
    "parse_crs_file",
    "parse_toronto_instance",
    "save_instance_npz",
    "summarize_instances",
    "validate_instance",
    "verify_checksums",
    "write_checksums",
]
