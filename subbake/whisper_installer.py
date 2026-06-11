"""whisper.cpp binary and model management.

Downloads and manages pre-built whisper-cli binaries from GitHub Releases
and GGML model files from Hugging Face.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


GITHUB_REPO = "ggerganov/whisper.cpp"
RELEASES_API_URL = "https://api.github.com/repos/{}/releases".format(GITHUB_REPO)
MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

SUPPORTED_MODELS = frozenset({"tiny", "base", "small", "medium", "large-v3"})
DEFAULT_MODEL = "small"

WHISPER_CLI_NAME = "whisper-cli.exe" if platform.system().lower() == "windows" else "whisper-cli"
LEGACY_BINARY_NAME = "main.exe" if platform.system().lower() == "windows" else "main"
BINARY_CANDIDATE_NAMES = (WHISPER_CLI_NAME, LEGACY_BINARY_NAME)


def _user_agent() -> str:
    return "subbake/1.0"


def _detect_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x64"
        raise RuntimeError("Linux arm64 whisper.cpp builds are not yet provided via pre-built releases")
    elif system == "darwin":
        if machine == "arm64":
            return "macos-arm64"
        elif machine in ("x86_64", "amd64"):
            return "macos-x64"
    elif system == "windows":
        if machine in ("x86_64", "amd64"):
            return "win-x64"

    raise RuntimeError(
        "Unsupported platform for pre-built whisper.cpp: {} {}".format(system, machine)
    )


def _archive_extension() -> str:
    return ".zip" if platform.system().lower() == "windows" else ".tar.gz"


def _platform_short(plat: str) -> str:
    """Map full platform string (e.g. ``linux-x64``) to short form (``x64``)."""
    mapping = {
        "linux-x64": "x64",
        "linux-aarch64": "aarch64",
        "macos-arm64": "arm64",
        "macos-x64": "x64",
        "win-x64": "x64",
        "win-x86": "Win32",
    }
    return mapping.get(plat, plat)


def _platform_keywords(plat: str) -> list[str]:
    """Return a list of keywords to identify assets for this platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    keywords = [machine]
    if system == "linux":
        if machine in ("x86_64", "amd64"):
            keywords = ["x64", "linux-x64"]
        elif machine in ("aarch64", "arm64"):
            keywords = ["aarch64", "arm64"]
    elif system == "darwin":
        if machine == "arm64":
            keywords = ["arm64", "macos-arm64"]
        else:
            keywords = ["x64", "macos-x64"]
    elif system == "windows":
        keywords = ["win-x64", "x64"] if "x86_64" in machine else ["Win32"]
    return keywords


def _splitext(path: str) -> tuple[str, str]:
    """Like os.path.splitext but handles ``.tar.gz`` as a single extension."""
    if path.endswith(".tar.gz"):
        return path[:-7], ".tar.gz"
    idx = path.rfind(".")
    if idx == -1:
        return path, ""
    return path[:idx], path[idx:]


def _is_zip_like(path: Path) -> bool:
    """Return True if the file looks like a ZIP archive."""
    # Check magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header[:2] == b"PK"  # ZIP magic
    except OSError:
        return False


def _extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract a zip or tar.gz archive into *dest*."""
    if archive_path.suffix == ".zip" or _is_zip_like(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
        return

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest)


def _download_file(
    url: str,
    dest: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Download *url* to *dest*, with optional progress reporting."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", "0")) or None
        downloaded = 0
        chunk_size = 65536
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total:
                    progress_callback(downloaded, total)


class WhisperInstaller:
    """Manages whisper.cpp binary and GGML model files under ``.subbake/whisper/``."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root / ".subbake" / "whisper"
        self._bin_dir = self._root / "bin"
        self._models_dir = self._root / "models"
        self._version_file = self._root / "version.txt"

    # ------------------------------------------------------------------
    # Binary management
    # ------------------------------------------------------------------

    @property
    def binary_path(self) -> Path:
        return self._bin_dir / WHISPER_CLI_NAME

    def is_available(self) -> bool:
        """Return True if whisper-cli is either managed or on PATH."""
        if self.binary_path.exists():
            return True
        return shutil.which(WHISPER_CLI_NAME) is not None

    def installed_version(self) -> str | None:
        """Read the installed version string, or None."""
        if not self._version_file.exists():
            return None
        return self._version_file.read_text().strip()

    def check_available(self) -> tuple[bool, str]:
        """Return (ok, message) describing whisper-cli availability."""
        if self.binary_path.exists():
            ver = self.installed_version()
            label = "whisper.cpp {}".format(ver) if ver else "whisper-cli"
            return True, "{} found at {}".format(label, self.binary_path)
        system_bin = shutil.which(WHISPER_CLI_NAME)
        if system_bin:
            return True, "whisper-cli found in PATH: {}".format(system_bin)
        return False, (
            "whisper-cli is not installed.\n"
            "  Install it:  sbake whisper install\n"
            "  Or see:      https://github.com/ggerganov/whisper.cpp"
        )

    def ensure_available(self) -> Path:
        """Return the path to whisper-cli, raising if unavailable."""
        ok, msg = self.check_available()
        if not ok:
            raise RuntimeError(msg)
        if self.binary_path.exists():
            return self.binary_path
        system_path = shutil.which(WHISPER_CLI_NAME)
        if system_path:
            return Path(system_path)
        raise RuntimeError("whisper-cli not found (unreachable)")

    # ------------------------------------------------------------------
    # Release management
    # ------------------------------------------------------------------

    def _release_info(self, version: str) -> dict:
        """Fetch release info from GitHub API."""
        if version == "latest":
            url = "{}/latest".format(RELEASES_API_URL)
        else:
            url = "{}/tags/{}".format(RELEASES_API_URL, version)
        req = urllib.request.Request(url, headers={"User-Agent": _user_agent(), "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError("Release {} not found on {}/releases".format(version, GITHUB_REPO)) from exc
            if exc.code == 403:
                raise RuntimeError(
                    "GitHub API rate limited. Try again later, or install whisper-cli manually."
                ) from exc
            raise RuntimeError("GitHub API error: {}".format(exc)) from exc

    def _find_asset(self, release: dict) -> dict:
        """Find the archive asset matching the current platform.

        Tries multiple naming patterns to handle whisper.cpp release
        format changes (old: ``whisper-cli-{tag}-{platform}.tar.gz``,
        new: ``whisper-bin-{platform}.zip``).
        """
        plat = _detect_platform()
        tag = release["tag_name"]

        # Build candidate names in priority order
        candidates: list[str] = []

        # Old naming: whisper-cli-{tag}-{platform}.tar.gz/zip
        candidates.append("whisper-cli-{}-{}.tar.gz".format(tag, plat))
        candidates.append("whisper-cli-{}-{}.zip".format(tag, plat))

        # New naming: whisper-bin-{short_platform}.zip
        short = _platform_short(plat)
        candidates.append("whisper-bin-{}.zip".format(short))
        candidates.append("whisper-{}.zip".format(short))

        # Also try with tag prefix (some releases include tag in name)
        candidates.append("whisper-{}-{}.zip".format(tag.lstrip("v"), short))

        # Try exact name match
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name in candidates:
                return asset

        # Fallback: match any asset whose name contains plat keywords
        plat_keywords = _platform_keywords(plat)
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if any(k in name for k in plat_keywords) and name.endswith(".zip"):
                return asset

        raise RuntimeError(
            "No pre-built binary found for {} in release {}. "
            "Assets available: {}".format(
                plat, tag,
                ", ".join(a.get("name", "?") for a in release.get("assets", [])),
            )
        )

    def _source_archive_url(self, release: dict) -> str:
        url = release.get("tarball_url")
        if url:
            return str(url)
        tag = str(release["tag_name"])
        return "https://github.com/{}/archive/refs/tags/{}.tar.gz".format(GITHUB_REPO, tag)

    def _find_extracted_binary(self, root: Path) -> Path | None:
        for name in BINARY_CANDIDATE_NAMES:
            direct = root / name
            if direct.exists() and direct.is_file():
                return direct
        for name in BINARY_CANDIDATE_NAMES:
            for candidate in root.rglob(name):
                if candidate.exists() and candidate.is_file():
                    return candidate
        return None

    def _promote_binary_directory(self, binary: Path) -> Path:
        """Move the binary and nearby runtime files to the managed bin dir."""
        source_dir = binary.parent
        if source_dir != self._bin_dir:
            for item in source_dir.iterdir():
                target = self._bin_dir / item.name
                if item == target:
                    continue
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                item.rename(target)

        managed = self._bin_dir / binary.name
        if managed.name != WHISPER_CLI_NAME:
            target = self.binary_path
            if target.exists():
                target.unlink()
            managed.rename(target)
            managed = target
        return managed

    def _finalize_binary_install(self, tag: str) -> Path:
        binary = self._find_extracted_binary(self._bin_dir)
        if binary is None:
            raise RuntimeError(
                "Downloaded whisper.cpp archive did not contain {}. "
                "Try installing whisper.cpp manually and ensure whisper-cli is on PATH.".format(WHISPER_CLI_NAME)
            )

        managed = self._promote_binary_directory(binary)
        managed.chmod(managed.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self._version_file.write_text(tag)
        print("whisper.cpp {} installed to {}".format(tag, managed), file=sys.stderr)
        return managed

    def _install_from_source(
        self,
        release: dict,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError(
                "No pre-built whisper.cpp binary is available for this platform, and cmake was not found. "
                "Install cmake and a C/C++ compiler, then rerun: sbake whisper install"
            )

        tag = str(release["tag_name"])
        source_archive = self._root / "source.tar.gz"
        source_root = self._root / "source"
        build_dir = self._root / "build"

        for path in (source_root, build_dir):
            if path.exists():
                shutil.rmtree(path)
        source_root.mkdir(parents=True, exist_ok=True)

        print("Downloading whisper.cpp {} source...".format(tag), file=sys.stderr)
        _download_file(self._source_archive_url(release), source_archive, progress_callback=progress_callback)
        try:
            print("Extracting source...", file=sys.stderr)
            _extract_archive(source_archive, source_root)
        finally:
            if source_archive.exists():
                source_archive.unlink()

        source_dirs = [p for p in source_root.iterdir() if p.is_dir()]
        source_dir = source_dirs[0] if len(source_dirs) == 1 else source_root

        configure_cmd = [
            cmake,
            "-S", str(source_dir),
            "-B", str(build_dir),
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_EXAMPLES=ON",
        ]
        build_cmd = [
            cmake,
            "--build", str(build_dir),
            "--config", "Release",
            "--target", "whisper-cli",
            "-j", str(max(1, os.cpu_count() or 1)),
        ]

        print("Configuring whisper.cpp build...", file=sys.stderr)
        subprocess.run(configure_cmd, check=True)
        print("Building whisper.cpp...", file=sys.stderr)
        try:
            subprocess.run(build_cmd, check=True)
        except subprocess.CalledProcessError:
            fallback_build_cmd = build_cmd[:6] + ["main"] + build_cmd[7:]
            subprocess.run(fallback_build_cmd, check=True)

        built_binary = self._find_extracted_binary(build_dir)
        if built_binary is None:
            raise RuntimeError("whisper.cpp build finished but no whisper-cli binary was produced.")

        self._bin_dir.mkdir(parents=True, exist_ok=True)
        target = self._bin_dir / built_binary.name
        if target.exists():
            target.unlink()
        shutil.copy2(built_binary, target)
        if target.name != WHISPER_CLI_NAME:
            renamed = self.binary_path
            if renamed.exists():
                renamed.unlink()
            target.rename(renamed)
            target = renamed

        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self._version_file.write_text(tag)
        shutil.rmtree(source_root, ignore_errors=True)
        print("whisper.cpp {} built and installed to {}".format(tag, target), file=sys.stderr)
        return target

    def latest_version(self) -> str:
        """Return the latest release tag name (e.g. \"v1.7.0\")."""
        return str(self._release_info("latest")["tag_name"])

    def install(
        self,
        version: str = "latest",
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download and install a specific whisper.cpp version.

        Returns the path to the installed binary.
        """
        if self._bin_dir.exists():
            shutil.rmtree(self._bin_dir)
        self._bin_dir.mkdir(parents=True, exist_ok=True)

        release = self._release_info(version)
        tag = release["tag_name"]
        try:
            asset = self._find_asset(release)
        except RuntimeError as asset_error:
            print("{} Falling back to source build.".format(asset_error), file=sys.stderr)
            return self._install_from_source(release, progress_callback=progress_callback)

        url = asset["browser_download_url"]

        # Use the URL's actual filename to determine extension
        url_name = asset.get("name", url.rsplit("/", 1)[-1])
        _, archive_ext = _splitext(url_name)
        archive_path = self._root / "download{}".format(archive_ext)

        print("Downloading whisper.cpp {}...".format(tag), file=sys.stderr)
        _download_file(url, archive_path, progress_callback=progress_callback)

        # Extract
        print("Extracting...", file=sys.stderr)
        try:
            _extract_archive(archive_path, self._bin_dir)
        finally:
            if archive_path.exists():
                archive_path.unlink()

        try:
            return self._finalize_binary_install(tag)
        except RuntimeError as install_error:
            print("{} Falling back to source build.".format(install_error), file=sys.stderr)
            shutil.rmtree(self._bin_dir, ignore_errors=True)
            self._bin_dir.mkdir(parents=True, exist_ok=True)
            return self._install_from_source(release, progress_callback=progress_callback)

    def update(
        self,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Update to the latest version, if newer."""
        try:
            latest = self.latest_version()
        except RuntimeError:
            raise RuntimeError("Could not check for updates. Try again later.") from None

        current = self.installed_version()
        if current and current == latest:
            print("whisper.cpp is already up to date ({})".format(latest), file=sys.stderr)
            return self.ensure_available()

        return self.install(latest, progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def model_path(self, model: str = DEFAULT_MODEL) -> Path:
        """Return the expected path for a GGML model file."""
        model = model.lower().strip()
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                "Unsupported model: {}. Supported: {}".format(model, ", ".join(sorted(SUPPORTED_MODELS)))
            )
        return self._models_dir / "ggml-{}.bin".format(model)

    def download_model(
        self,
        model: str = DEFAULT_MODEL,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download a GGML model file from Hugging Face.

        Returns the path to the downloaded model.
        """
        model = model.lower().strip()
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                "Unsupported model: {}. Supported: {}".format(model, ", ".join(sorted(SUPPORTED_MODELS)))
            )

        dest = self.model_path(model)
        dest.parent.mkdir(parents=True, exist_ok=True)

        url = "{}/ggml-{}.bin".format(MODEL_BASE_URL, model)
        print("Downloading model ggml-{}.bin...".format(model), file=sys.stderr)
        _download_file(url, dest, progress_callback=progress_callback)

        print("Model saved to {}".format(dest), file=sys.stderr)
        return dest

    def ensure_model(
        self,
        model: str = DEFAULT_MODEL,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Return a model path, downloading if needed."""
        path = self.model_path(model)
        if path.exists():
            return path
        return self.download_model(model, progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # Additional management
    # ------------------------------------------------------------------

    def uninstall(self, *, keep_models: bool = False) -> None:
        """Remove managed whisper.cpp data.

        When ``keep_models`` is True, only the managed binary/build files and
        version metadata are removed.
        """
        if not self._root.exists():
            print("whisper.cpp is not installed (no data found).", file=sys.stderr)
            return

        if keep_models:
            for path in (self._bin_dir, self._root / "build", self._root / "source"):
                if path.exists():
                    shutil.rmtree(path)
            if self._version_file.exists():
                self._version_file.unlink()
            print("whisper.cpp binary removed from {} (models kept).".format(self._root), file=sys.stderr)
            return

        shutil.rmtree(self._root)
        print("whisper.cpp removed from {}".format(self._root), file=sys.stderr)

    def list_versions(self) -> list[str]:
        """Return a list of installed version tags."""
        versions: list[str] = []
        current = self.installed_version()
        if current:
            versions.append(current)
        return versions

    def list_models(self) -> list[dict[str, object]]:
        """Return info about downloaded models."""
        models: list[dict[str, object]] = []
        if not self._models_dir.exists():
            return models
        for f in sorted(self._models_dir.iterdir()):
            if f.is_file() and f.name.startswith("ggml-") and f.suffix == ".bin":
                model_name = f.name.replace("ggml-", "").replace(".bin", "")
                models.append({
                    "name": model_name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                })
        return models

    def remove_model(self, model: str) -> bool:
        """Remove a specific downloaded model. Returns True if removed."""
        path = self.model_path(model)
        if path.exists():
            path.unlink()
            print("Removed model: {}".format(path), file=sys.stderr)
            return True
        print("Model {} not found.".format(model), file=sys.stderr)
        return False
