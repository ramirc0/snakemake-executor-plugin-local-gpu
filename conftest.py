import os

import pytest

STUB_SCRIPT = """\
#!/bin/sh
dir=$(dirname "$0")
printf '%s\\t%s\\n' "${CUDA_VISIBLE_DEVICES-<unset>}" "$*" >> "$dir/calls"
case "$1" in
    --query-gpu=index,uuid) cat "$dir/gpus" ;;
    --query-compute-apps=gpu_uuid) cat "$dir/busy" ;;
    *) echo "stub nvidia-smi: unexpected arguments: $*" >&2; exit 2 ;;
esac
"""


class StubNvidiaSmi:
    """An `nvidia-smi` on PATH that answers from files the test controls."""

    def __init__(self, directory):
        self.directory = directory
        self.directory.mkdir()
        script = self.directory / "nvidia-smi"
        script.write_text(STUB_SCRIPT)
        script.chmod(0o755)
        (self.directory / "calls").touch()
        self.reports_gpus("GPU-aaaa", "GPU-bbbb")
        self.reports_busy()

    def reports_gpus(self, *uuids):
        lines = (f"{index}, {uuid}\n" for index, uuid in enumerate(uuids))
        (self.directory / "gpus").write_text("".join(lines))

    def reports_busy(self, *uuids):
        (self.directory / "busy").write_text("".join(f"{uuid}\n" for uuid in uuids))

    @property
    def calls(self):
        """Each call as a (caller's CUDA_VISIBLE_DEVICES, arguments) pair."""
        lines = (self.directory / "calls").read_text().splitlines()
        return [tuple(line.split("\t")) for line in lines]


@pytest.fixture(autouse=True)
def nvidia_smi(tmp_path, monkeypatch):
    """Put a stub nvidia-smi with two idle GPUs first on PATH.

    Every test and doctest gets it, so none can reach the host's real one.
    """
    stub = StubNvidiaSmi(tmp_path / "nvidia-smi-stub")
    monkeypatch.setenv("PATH", f"{stub.directory}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return stub
