"""Verify registry artifacts against the release build, then test a clean install."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


INDEX_URLS = {"testpypi": "https://test.pypi.org", "pypi": "https://pypi.org"}
FILE_HOSTS = {"files.pythonhosted.org", "test-files.pythonhosted.org"}
FORMAT = "nyssa-published-release-verification-v1"
VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:rc(?:0|[1-9][0-9]*))?")


class ReleaseNotReady(ValueError):
    """The registry has not exposed both uploaded distributions yet."""


def _trusted_url(url: str, hosts: set[str]) -> None:
    if not isinstance(url, str):
        raise ValueError("registry URL must be a string")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in hosts
            or parsed.username or parsed.password or parsed.port not in {None, 443}
            or parsed.fragment):
        raise ValueError(f"untrusted registry URL: {url}")


class _RegistryRedirect(HTTPRedirectHandler):
    def __init__(self, hosts: set[str]) -> None:
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _trusted_url(newurl, self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url: str, hosts: set[str]):
    _trusted_url(url, hosts)
    request = Request(url, headers={"User-Agent": "nyssa-bench-release-verification"})
    return build_opener(_RegistryRedirect(hosts)).open(request, timeout=30)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_artifacts(dist_dir: Path, version: str) -> dict[str, dict[str, Any]]:
    wheel = dist_dir / f"nyssa_bench-{version}-py3-none-any.whl"
    sdist = dist_dir / f"nyssa_bench-{version}.tar.gz"
    expected = {}
    for path, kind in ((wheel, "bdist_wheel"), (sdist, "sdist")):
        if not path.is_file():
            raise ValueError(f"missing expected build artifact: {path.name}")
        expected[path.name] = {"sha256": _sha256(path), "size": path.stat().st_size, "packagetype": kind}
    return expected


def _select_files(payload: dict[str, Any], version: str, expected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if (not isinstance(payload, dict) or not isinstance(payload.get("info"), dict)
            or not isinstance(payload.get("urls"), list)
            or not all(isinstance(item, dict) for item in payload["urls"])):
        raise ValueError("registry metadata must contain an info mapping and a urls list")
    info = payload.get("info", {})
    name = info.get("name")
    if (not isinstance(name, str) or re.sub(r"[-_.]+", "-", name.lower()) != "nyssa-bench"
            or info.get("version") != version):
        raise ValueError("registry package identity does not match the requested release")
    selected = []
    for filename, build in expected.items():
        matches = [item for item in payload.get("urls", []) if item.get("filename") == filename]
        if not matches:
            raise ReleaseNotReady(f"registry has not exposed {filename}")
        if len(matches) != 1:
            raise ValueError(f"registry returned duplicate records for {filename}")
        item = matches[0]
        if item.get("yanked"):
            raise ValueError(f"published artifact is yanked: {filename}")
        if item.get("packagetype") != build["packagetype"]:
            raise ValueError(f"published artifact type differs from build: {filename}")
        digests = item.get("digests")
        if (not isinstance(digests, dict) or digests.get("sha256") != build["sha256"]
                or item.get("size") != build["size"]):
            raise ValueError(f"published artifact differs from the release build: {filename}")
        _trusted_url(item.get("url"), FILE_HOSTS)
        selected.append(item)
    return selected


def _published_files(index: str, version: str, expected: dict[str, dict[str, Any]],
                     attempts: int, retry_delay: float) -> list[dict[str, Any]]:
    url = f"{INDEX_URLS[index]}/pypi/nyssa-bench/{version}/json"
    for attempt in range(attempts):
        try:
            with _open(url, {urlsplit(INDEX_URLS[index]).hostname}) as response:
                body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("registry metadata exceeds 2 MiB")
            return _select_files(json.loads(body), version, expected)
        except (ReleaseNotReady, HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {404, 429, 500, 502, 503, 504}:
                raise
            if attempt + 1 == attempts:
                raise
            print(f"Waiting for {index} release visibility ({attempt + 1}/{attempts}): {exc}", flush=True)
            time.sleep(retry_delay)
    raise AssertionError("unreachable")


def _download(item: dict[str, Any], expected: dict[str, Any], destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    with _open(item["url"], FILE_HOSTS) as response, partial.open("wb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            size += len(block)
            if size > expected["size"]:
                raise ValueError(f"download exceeds build artifact size: {destination.name}")
            digest.update(block)
            handle.write(block)
    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"download hash or size mismatch: {destination.name}")
    partial.replace(destination)


def _run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> str:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("+ " + subprocess.list2cmdline(command) + "\n")
        handle.flush()
        try:
            result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=600)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            handle.write(output + "\nCommand timed out after 600 seconds.\n")
            raise
        handle.write(result.stdout + "\n")
    if result.returncode:
        raise RuntimeError(f"installed-release command failed ({result.returncode}); see {log.name}")
    return result.stdout.strip()


def _install_smoke(wheel: Path, out: Path, version: str) -> dict[str, Any]:
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME"} and not key.startswith("PIP_")}
    env.update({"PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull})
    log = out / "installation.log"
    with tempfile.TemporaryDirectory(prefix="nyssa-published-install-") as temporary:
        workspace = Path(temporary).resolve()
        venv = workspace / "venv"
        _run([sys.executable, "-I", "-m", "venv", str(venv)], workspace, env, log)
        scripts = venv / ("Scripts" if sys.platform == "win32" else "bin")
        python = scripts / ("python.exe" if sys.platform == "win32" else "python")
        nyssa = scripts / ("nyssa.exe" if sys.platform == "win32" else "nyssa")
        _run([str(python), "-I", "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
              "--disable-pip-version-check", str(wheel.resolve())], workspace, env, log)
        _run([str(python), "-I", "-m", "pip", "check"], workspace, env, log)
        _run([str(nyssa), "--help"], workspace, env, log)
        observed = _run([str(nyssa), "--version"], workspace, env, log)
        if observed != f"nyssa {version}":
            raise ValueError(f"installed CLI version differs: {observed!r}")
        probe = _run([str(python), "-I", "-c",
                      "import importlib.metadata,json,nyssa_bench; print(json.dumps({"
                      "'api_version':nyssa_bench.__version__,"
                      "'distribution_version':importlib.metadata.version('nyssa-bench')}))"],
                     workspace, env, log)
        versions = json.loads(probe)
        if versions != {"api_version": version, "distribution_version": version}:
            raise ValueError("installed API and distribution versions differ from the release")
        for command in ("list-suites", "list-stressors"):
            _run([str(nyssa), command], workspace, env, log)
        _run([str(python), "-I", "-m", "nyssa_bench.packaging_smoke", "--out", str(out / "smoke")],
             workspace, env, log)
    return {"status": "passed", **versions, "cli_version": observed,
            "dependency_index": "https://pypi.org/simple", "source_checkout_used": False}


def verify_published_release(*, version: str, dist_dir: Path, out_dir: Path,
                             index: str = "testpypi", attempts: int = 12,
                             retry_delay: float = 5.0) -> dict[str, Any]:
    if not VERSION.fullmatch(version) or index not in INDEX_URLS:
        raise ValueError("unsupported release version or package index")
    if attempts < 1 or not math.isfinite(retry_delay) or retry_delay < 0:
        raise ValueError("attempts must be positive and retry_delay finite and non-negative")
    out = out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("verification output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"format": FORMAT, "package": "nyssa-bench", "version": version,
                              "index": INDEX_URLS[index], "python_version": sys.version.split()[0],
                              "status": "failed", "artifacts": []}
    try:
        expected = _expected_artifacts(dist_dir.resolve(), version)
        files = _published_files(index, version, expected, attempts, retry_delay)
        downloads = out / "distributions"
        downloads.mkdir()
        for item in files:
            name = item["filename"]
            _download(item, expected[name], downloads / name)
            report["artifacts"].append({"filename": name, "url": item["url"], **expected[name]})
        wheel = next(downloads / item["filename"] for item in files if item["packagetype"] == "bdist_wheel")
        report["installation"] = _install_smoke(wheel, out, version)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (out / "published_release_verification.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", choices=tuple(INDEX_URLS), default="testpypi")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        report = verify_published_release(version=args.version, dist_dir=args.dist_dir, out_dir=args.out,
                                          index=args.index, attempts=args.attempts, retry_delay=args.retry_delay)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"published-release verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
