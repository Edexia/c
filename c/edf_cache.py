"""Hash-based EDF file lookup with persistent JSON cache."""

import json
from pathlib import Path
from typing import Optional

from edf import EDF

CACHE_FILENAME = ".edf_cache.json"


def compute_edf_content_hash(edf_path: Path) -> Optional[str]:
    """Compute the content hash of an EDF file using the EDF SDK."""
    try:
        with EDF.open(edf_path) as edf:
            return edf.content_hash
    except Exception:
        pass
    return None


def load_cache(cache_path: Path) -> dict[str, str]:
    """Load the hash->filename cache from disk."""
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Save the hash->filename cache to disk."""
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass


def scan_edf_files(directory: Path) -> list[Path]:
    """Recursively scan a directory for all EDF files."""
    edf_files = []
    if not directory.exists():
        return edf_files

    # Find all .edf files recursively
    edf_files.extend(directory.rglob("*.edf"))

    # Also find unpacked EDF directories (with manifest.json)
    for manifest in directory.rglob("manifest.json"):
        parent = manifest.parent
        # Check it looks like an EDF directory (has submissions folder)
        if (parent / "submissions").is_dir():
            edf_files.append(parent)

    return list(set(edf_files))


def build_hash_index(edf_files: list[Path]) -> dict[str, Path]:
    """Build an index mapping content_hash -> Path for a list of EDF files."""
    index: dict[str, Path] = {}
    for edf_path in edf_files:
        content_hash = compute_edf_content_hash(edf_path)
        if content_hash:
            index[content_hash] = edf_path
    return index


class EDFCache:
    """Cache for finding EDF files by their content hash."""

    def __init__(self, edf_directory: Path):
        self.directory = edf_directory
        self.cache_path = edf_directory / CACHE_FILENAME
        self._cache: dict[str, str] = {}
        self._hash_to_path: dict[str, Path] = {}
        self._loaded = False

    def _load(self) -> None:
        """Lazily load cache and scan directory."""
        if self._loaded:
            return

        self._cache = load_cache(self.cache_path)
        self._hash_to_path = {}

        edf_files = scan_edf_files(self.directory)

        for edf_path in edf_files:
            filename = edf_path.name

            if filename in self._cache:
                content_hash = self._cache[filename]
                self._hash_to_path[content_hash] = edf_path
            else:
                content_hash = compute_edf_content_hash(edf_path)
                if content_hash:
                    self._cache[filename] = content_hash
                    self._hash_to_path[content_hash] = edf_path

        save_cache(self.cache_path, self._cache)
        self._loaded = True

    def find_by_hash(self, content_hash: str) -> Optional[Path]:
        """Find an EDF file by its content hash."""
        self._load()
        return self._hash_to_path.get(content_hash)

    def validate_cache(self) -> list[str]:
        """Re-scan directory and return list of missing files."""
        self._cache = load_cache(self.cache_path)
        edf_files = scan_edf_files(self.directory)
        current_filenames = {p.name for p in edf_files}

        missing = []
        for filename in list(self._cache.keys()):
            if filename not in current_filenames:
                missing.append(filename)
                del self._cache[filename]

        if missing:
            save_cache(self.cache_path, self._cache)

        self._loaded = False
        return missing

    def get_all_hashes(self) -> list[str]:
        """Get all content hashes in the cache."""
        self._load()
        return list(self._hash_to_path.keys())

    def refresh(self) -> None:
        """Force refresh the cache."""
        self._loaded = False
        self._cache = {}
        self._hash_to_path = {}
        self._load()
