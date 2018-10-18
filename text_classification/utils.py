from pathlib import Path


def is_empty(folder: Path):
    non_empty_dirs = {str(p.parent) for p in folder.rglob('*') if p.is_file()}
    return not non_empty_dirs
