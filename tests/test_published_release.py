from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from scripts import verify_published_release as verifier


VERSION = "0.1.0rc1"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    contents = {
        f"nyssa_bench-{VERSION}-py3-none-any.whl": b"wheel contents",
        f"nyssa_bench-{VERSION}.tar.gz": b"source archive contents",
    }
    for name, data in contents.items():
        (dist / name).write_bytes(data)
    expected = verifier._expected_artifacts(dist, VERSION)
    payload = {
        "info": {"name": "nyssa-bench", "version": VERSION},
        "urls": [
            {"filename": name, "packagetype": item["packagetype"], "size": item["size"],
             "digests": {"sha256": item["sha256"]}, "yanked": False,
             "url": f"https://files.pythonhosted.org/packages/{name}"}
            for name, item in expected.items()
        ],
    }

    def open_registry(url, hosts):
        verifier._trusted_url(url, hosts)
        if url.endswith("/json"):
            return BytesIO(json.dumps(payload).encode())
        return BytesIO(contents[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(verifier, "_open", open_registry)
    return dist, expected, payload, contents


@pytest.mark.parametrize("index", ["testpypi", "pypi"])
def test_downloads_both_matching_artifacts_before_install(registry, tmp_path, monkeypatch, index):
    dist, expected, _, contents = registry
    out = tmp_path / "result"

    def install(wheel, output, version):
        assert version == VERSION
        assert wheel.parent == output / "distributions"
        for name, data in contents.items():
            assert (wheel.parent / name).read_bytes() == data
        return {"status": "passed"}

    monkeypatch.setattr(verifier, "_install_smoke", install)
    report = verifier.verify_published_release(version=VERSION, dist_dir=dist, out_dir=out, index=index)
    assert report["status"] == "passed"
    assert report["index"] == verifier.INDEX_URLS[index]
    assert {item["filename"]: item["sha256"] for item in report["artifacts"]} == {
        name: item["sha256"] for name, item in expected.items()
    }
    assert json.loads((out / "published_release_verification.json").read_text()) == report


@pytest.mark.parametrize("change, message", [
    ({"yanked": True}, "yanked"),
    ({"packagetype": "sdist"}, "type differs"),
    ({"digests": {"sha256": "0" * 64}}, "differs from the release build"),
    ({"digests": None}, "differs from the release build"),
    ({"size": 9999}, "differs from the release build"),
    ({"url": "https://attacker.example/file.whl"}, "untrusted registry URL"),
])
def test_rejects_invalid_registry_artifacts(registry, change, message):
    _, expected, payload, _ = registry
    payload["urls"][0].update(change)
    with pytest.raises(ValueError, match=message):
        verifier._select_files(payload, VERSION, expected)


@pytest.mark.parametrize("info", [None, {}, {"name": None}, {"name": "another-project", "version": VERSION},
                                  {"name": "nyssa-bench", "version": "0.1.0"}])
def test_rejects_wrong_or_malformed_identity(registry, info):
    _, expected, payload, _ = registry
    payload["info"] = info
    with pytest.raises(ValueError):
        verifier._select_files(payload, VERSION, expected)


def test_rejects_duplicate_registry_records(registry):
    _, expected, payload, _ = registry
    payload["urls"].append(copy.deepcopy(payload["urls"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        verifier._select_files(payload, VERSION, expected)


def test_retries_visibility_until_both_files_are_available(registry, monkeypatch):
    _, expected, payload, _ = registry
    incomplete = copy.deepcopy(payload)
    incomplete["urls"].pop()
    replies = iter([HTTPError("https://test.pypi.org", 404, "not yet", {}, None), incomplete, payload])
    sleeps = []

    def open_registry(url, hosts):
        response = next(replies)
        if isinstance(response, Exception):
            raise response
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(verifier, "_open", open_registry)
    monkeypatch.setattr(verifier.time, "sleep", sleeps.append)
    assert len(verifier._published_files("testpypi", VERSION, expected, 3, 5)) == 2
    assert sleeps == [5, 5]


@pytest.mark.parametrize("failure", ["missing", "digest", "forbidden"])
def test_registry_failure_is_bounded_and_retained(registry, tmp_path, monkeypatch, failure):
    dist, _, payload, _ = registry
    sleeps = []
    monkeypatch.setattr(verifier.time, "sleep", sleeps.append)
    if failure == "missing":
        payload["urls"].pop()
    elif failure == "digest":
        payload["urls"][0]["digests"]["sha256"] = "0" * 64
    else:
        def forbidden(*args):
            raise HTTPError("https://test.pypi.org", 403, "forbidden", {}, None)
        monkeypatch.setattr(verifier, "_open", forbidden)
    out = tmp_path / "failed"
    with pytest.raises((ValueError, HTTPError)):
        verifier.verify_published_release(version=VERSION, dist_dir=dist, out_dir=out, attempts=2, retry_delay=0)
    report = json.loads((out / "published_release_verification.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]
    assert "installation" not in report
    assert sleeps == ([0] if failure == "missing" else [])


@pytest.mark.parametrize("url", [
    "http://files.pythonhosted.org/file", "https://files.pythonhosted.org.attacker.example/file",
    "https://user:password@files.pythonhosted.org/file", "https://files.pythonhosted.org:444/file",
    "https://files.pythonhosted.org/file#fragment", "file:///tmp/wheel.whl",
])
def test_rejects_untrusted_downloads_and_redirects(url):
    with pytest.raises(ValueError, match="untrusted registry URL"):
        verifier._trusted_url(url, verifier.FILE_HOSTS)
    redirect = verifier._RegistryRedirect(verifier.FILE_HOSTS)
    with pytest.raises(ValueError, match="untrusted registry URL"):
        redirect.redirect_request(Request("https://files.pythonhosted.org/file"), None, 302, "Found", {}, url)


@pytest.mark.parametrize("replacement", [b"", b"wrong archive contents!", b"x" * 100])
def test_corrupt_sdist_blocks_install_and_retains_failure(registry, tmp_path, monkeypatch, replacement):
    dist, _, _, contents = registry
    contents[f"nyssa_bench-{VERSION}.tar.gz"] = replacement
    monkeypatch.setattr(verifier, "_install_smoke", lambda *args: pytest.fail("unverified artifact installed"))
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="size|hash"):
        verifier.verify_published_release(version=VERSION, dist_dir=dist, out_dir=out)
    report = json.loads((out / "published_release_verification.json").read_text())
    assert report["status"] == "failed"
    assert len(report["artifacts"]) == 1
    assert not (out / "distributions" / f"nyssa_bench-{VERSION}.tar.gz").exists()


@pytest.mark.parametrize("bad_version", ["cli", "api", "metadata", None])
def test_install_uses_isolated_environment_and_checks_all_versions(tmp_path, monkeypatch, bad_version):
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://attacker.example/simple")
    wheel = tmp_path / "release.whl"
    commands = []

    def run(command, cwd, env, log):
        assert cwd != Path.cwd() and not cwd.is_relative_to(Path.cwd())
        assert "PYTHONPATH" not in env and "PIP_EXTRA_INDEX_URL" not in env
        assert env["PIP_CONFIG_FILE"] == verifier.os.devnull
        assert env["PYTHONNOUSERSITE"] == "1"
        commands.append(command)
        if "--version" in command:
            return "nyssa " + ("9.9.9" if bad_version == "cli" else VERSION)
        if "-c" in command:
            return json.dumps({"api_version": "9.9.9" if bad_version == "api" else VERSION,
                               "distribution_version": "9.9.9" if bad_version == "metadata" else VERSION})
        return ""

    monkeypatch.setattr(verifier, "_run", run)
    if bad_version:
        with pytest.raises(ValueError, match="version"):
            verifier._install_smoke(wheel, tmp_path, VERSION)
    else:
        assert verifier._install_smoke(wheel, tmp_path, VERSION)["status"] == "passed"
        assert any("nyssa_bench.packaging_smoke" in cmd for cmd in commands)
        assert any(cmd[-1] == "check" for cmd in commands)
    install = next(cmd for cmd in commands if "install" in cmd)
    assert install[-1] == str(wheel.resolve())
    assert install[install.index("--index-url") + 1] == "https://pypi.org/simple"
    assert "--extra-index-url" not in install


def test_timeout_preserves_command_output(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 600, output=b"download interrupted")
    monkeypatch.setattr(subprocess, "run", timeout)
    log = tmp_path / "installation.log"
    with pytest.raises(subprocess.TimeoutExpired):
        verifier._run(["python", "--version"], tmp_path, {}, log)
    assert "download interrupted" in log.read_text()
    assert "timed out" in log.read_text()


def test_installer_failure_report_is_retained(registry, tmp_path, monkeypatch):
    dist, _, _, _ = registry
    def fail(*args):
        raise RuntimeError("CLI smoke failed")
    monkeypatch.setattr(verifier, "_install_smoke", fail)
    out = tmp_path / "failed"
    assert verifier.main(["--version", VERSION, "--dist-dir", str(dist), "--out", str(out)]) == 1
    report = json.loads((out / "published_release_verification.json").read_text())
    assert report["status"] == "failed"
    assert report["error"] == "CLI smoke failed"
    assert len(report["artifacts"]) == 2


def test_refuses_to_overwrite_existing_evidence(registry, tmp_path):
    dist, _, _, _ = registry
    out = tmp_path / "existing"
    out.mkdir()
    evidence = out / "published_release_verification.json"
    evidence.write_text("previous run")
    with pytest.raises(ValueError, match="must be empty"):
        verifier.verify_published_release(version=VERSION, dist_dir=dist, out_dir=out)
    assert evidence.read_text() == "previous run"


def test_download_checks_bytes_even_when_registry_digest_matches(registry, tmp_path, monkeypatch):
    _, expected, payload, _ = registry
    item = payload["urls"][0]
    bad_bytes = b"X" * expected[item["filename"]]["size"]
    assert hashlib.sha256(bad_bytes).hexdigest() != item["digests"]["sha256"]
    monkeypatch.setattr(verifier, "_open", lambda *args: BytesIO(bad_bytes))
    with pytest.raises(ValueError, match="hash or size mismatch"):
        verifier._download(item, expected[item["filename"]], tmp_path / item["filename"])
