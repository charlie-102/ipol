"""Asset management for large model weights.

Manages downloading and caching of large model files (weights, datasets)
from cloud storage (HuggingFace Hub) to keep the main repo lightweight.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# Default cache directory - try home first, fall back to /tmp
import os
_home_cache = Path.home() / ".ipol_cache"
_tmp_cache = Path("/tmp/claude/.ipol_cache")
try:
    _home_cache.mkdir(parents=True, exist_ok=True)
    DEFAULT_CACHE_DIR = _home_cache
except (PermissionError, OSError):
    _tmp_cache.mkdir(parents=True, exist_ok=True)
    DEFAULT_CACHE_DIR = _tmp_cache

# Asset manifest file
MANIFEST_FILE = Path(__file__).parent.parent / "assets.json"


@dataclass
class Asset:
    """Represents a downloadable asset."""
    key: str  # Unique key in manifest
    local_path: str  # Relative path in method directory
    size: int  # Size in bytes
    sha256: str  # SHA256 checksum
    remote_url: Optional[str] = None  # Direct URL if not from HuggingFace


class AssetManager:
    """Manages downloading and caching of large assets."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        offline: bool = False
    ):
        """Initialize asset manager.

        Args:
            cache_dir: Directory to cache downloaded assets
            manifest_path: Path to assets.json manifest
            offline: If True, only use cached assets
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = manifest_path or MANIFEST_FILE
        self.manifest = self._load_manifest()
        self.offline = offline

    def _load_manifest(self) -> dict:
        """Load asset manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"version": "1.0", "provider": "huggingface", "repo_id": "", "assets": {}}

    def save_manifest(self):
        """Save asset manifest."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def get_asset(self, key: str) -> Optional[Asset]:
        """Get asset by key from manifest."""
        if key not in self.manifest.get("assets", {}):
            return None

        info = self.manifest["assets"][key]
        return Asset(
            key=key,
            local_path=info.get("local_path", key),
            size=info.get("size", 0),
            sha256=info.get("sha256", ""),
            remote_url=info.get("remote_url")
        )

    def get_cache_path(self, key: str) -> Path:
        """Get cache path for an asset."""
        return self.cache_dir / key.replace("/", "_")

    def is_cached(self, key: str) -> bool:
        """Check if asset is cached."""
        cache_path = self.get_cache_path(key)
        return cache_path.exists()

    def verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """Verify file checksum."""
        if not expected_sha256:
            return True

        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_sha256

    def download_asset(self, key: str, force: bool = False) -> Optional[Path]:
        """Download an asset.

        Args:
            key: Asset key from manifest
            force: If True, download even if cached

        Returns:
            Path to downloaded file, or None if failed
        """
        asset = self.get_asset(key)
        if not asset:
            print(f"Asset not found in manifest: {key}")
            return None

        cache_path = self.get_cache_path(key)

        # Check cache
        if cache_path.exists() and not force:
            if self.verify_checksum(cache_path, asset.sha256):
                return cache_path
            print(f"Cached file checksum mismatch, re-downloading: {key}")

        if self.offline:
            print(f"Offline mode: cannot download {key}")
            return None

        # Download from provider
        provider = self.manifest.get("provider", "huggingface")

        try:
            if provider == "huggingface":
                return self._download_from_huggingface(key, asset, cache_path)
            elif asset.remote_url:
                return self._download_from_url(asset.remote_url, cache_path, asset.sha256)
            else:
                print(f"No download source for asset: {key}")
                return None
        except Exception as e:
            print(f"Failed to download {key}: {e}")
            return None

    def _download_from_huggingface(self, key: str, asset: Asset, cache_path: Path) -> Optional[Path]:
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return None

        repo_id = self.manifest.get("repo_id", "")
        if not repo_id:
            print("No HuggingFace repo_id in manifest")
            return None

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=key,
                cache_dir=str(self.cache_dir / "hf_cache")
            )
            # Copy to our cache location
            import shutil
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded_path, cache_path)

            if self.verify_checksum(cache_path, asset.sha256):
                return cache_path
            else:
                print(f"Downloaded file checksum mismatch: {key}")
                cache_path.unlink()
                return None
        except Exception as e:
            print(f"HuggingFace download failed: {e}")
            return None

    def _download_from_url(self, url: str, cache_path: Path, sha256: str) -> Optional[Path]:
        """Download from direct URL."""
        import urllib.request

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, cache_path)

        if self.verify_checksum(cache_path, sha256):
            return cache_path
        else:
            print("Downloaded file checksum mismatch")
            cache_path.unlink()
            return None

    def ensure_asset(self, key: str, target_path: Path) -> bool:
        """Ensure asset exists at target path.

        Downloads if not cached, then copies/links to target.

        Args:
            key: Asset key
            target_path: Where the asset should be

        Returns:
            True if asset is ready at target_path
        """
        if target_path.exists():
            asset = self.get_asset(key)
            if asset and self.verify_checksum(target_path, asset.sha256):
                return True

        cache_path = self.download_asset(key)
        if not cache_path:
            return False

        # Copy to target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(cache_path, target_path)
        return True

    def list_assets(self) -> List[Dict]:
        """List all assets in manifest."""
        result = []
        for key, info in self.manifest.get("assets", {}).items():
            cached = self.is_cached(key)
            result.append({
                "key": key,
                "local_path": info.get("local_path", key),
                "size": info.get("size", 0),
                "size_mb": info.get("size", 0) / (1024 * 1024),
                "cached": cached
            })
        return result

    def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear cached assets.

        Args:
            older_than_days: Only clear files older than N days
        """
        import time

        now = time.time()
        cleared = 0
        cleared_bytes = 0

        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                if older_than_days:
                    age_days = (now - cache_file.stat().st_mtime) / 86400
                    if age_days < older_than_days:
                        continue

                size = cache_file.stat().st_size
                cache_file.unlink()
                cleared += 1
                cleared_bytes += size

        print(f"Cleared {cleared} files ({cleared_bytes / 1024 / 1024:.1f} MB)")

    def get_status(self) -> Dict:
        """Get cache status."""
        total_size = 0
        cached_count = 0
        total_count = 0

        for key in self.manifest.get("assets", {}):
            total_count += 1
            if self.is_cached(key):
                cached_count += 1
                total_size += self.get_cache_path(key).stat().st_size

        return {
            "cache_dir": str(self.cache_dir),
            "total_assets": total_count,
            "cached_assets": cached_count,
            "cache_size_mb": total_size / (1024 * 1024)
        }


def print_asset_status():
    """Print asset status report."""
    manager = AssetManager()
    status = manager.get_status()

    print("Asset Cache Status")
    print("=" * 40)
    print(f"Cache directory: {status['cache_dir']}")
    print(f"Cached: {status['cached_assets']}/{status['total_assets']} assets")
    print(f"Cache size: {status['cache_size_mb']:.1f} MB")
    print()

    assets = manager.list_assets()
    if assets:
        print("Assets:")
        for asset in sorted(assets, key=lambda a: a['key']):
            status = "[OK]" if asset['cached'] else "[--]"
            print(f"  {status} {asset['key']} ({asset['size_mb']:.1f} MB)")
    else:
        print("No assets in manifest")


if __name__ == "__main__":
    print_asset_status()
