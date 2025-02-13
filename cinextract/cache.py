from pathlib import Path
import hashlib
import json


CACHE_DIR = Path("~/.cache/cinextract").expanduser()


def get_cache_path_from(
    namesake: Path,
    extension: str = "",
    **invalidating_options: str | int | float,
) -> Path:
    """Create a cache path based on a related path and arbitrary arguments."""
    options_str = json.dumps(
        invalidating_options,
        sort_keys=True,
        ensure_ascii=True,
    )
    cache_key = hashlib.blake2b(
        f"{namesake.absolute()}{options_str}".encode(),
        digest_size=8,
    ).hexdigest()
    
    ext = f".{extension.lstrip('.')}" if extension else ""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{namesake.stem}_{cache_key}{ext}"
