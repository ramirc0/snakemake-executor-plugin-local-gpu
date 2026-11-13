import fcntl
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

GPU_JOB = """\
echo $CUDA_VISIBLE_DEVICES $(date +%s.%N) > {output}
sleep 2
date +%s.%N >> {output}
"""


@pytest.fixture
def workdir(tmp_path):
    workdir = tmp_path / "workflow"
    workdir.mkdir()
    return workdir


def run_snakemake(workdir, snakefile, *args, succeeds=True):
    (workdir / "Snakefile").write_text(textwrap.dedent(snakefile))
    command = [
        sys.executable,
        "-m",
        "snakemake",
        "--executor",
        "local-gpu",
        "--cores",
        "8",
        "--local-gpu-lock-dir",
        str(workdir / "locks"),
        "--local-gpu-poll-interval",
        "1",
        *args,
    ]
    driver = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log, _ = driver.communicate(timeout=120)
    assert (driver.returncode == 0) == succeeds, log
    return SimpleNamespace(pid=driver.pid, log=log)


def gpu_run(path):
    """Read the UUIDs, start and end time that GPU_JOB wrote."""
    first, end = path.read_text().splitlines()
    uuids, start = first.split()
    return SimpleNamespace(uuids=uuids.split(","), start=float(start), end=float(end))


def test_gpu_jobs_run_at_once_on_distinct_gpus(workdir, nvidia_smi):
    uuids = ["GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd"]
    nvidia_smi.reports_gpus(*uuids)
    run_snakemake(
        workdir,
        f"""
        rule all:
            input: expand("out/{{i}}.txt", i=range(3))

        rule gpu:
            output: "out/{{i}}.txt"
            resources: gpu=1
            shell: {GPU_JOB!r}
        """,
        "--resources",
        "gpu=3",
    )
    runs = [gpu_run(workdir / f"out/{i}.txt") for i in range(3)]
    leased = [uuid for run in runs for uuid in run.uuids]
    assert len(set(leased)) == 3
    assert set(leased) <= set(uuids)
    assert max(run.start for run in runs) < min(run.end for run in runs)


def test_job_waits_for_a_released_gpu(workdir, nvidia_smi):
    run = run_snakemake(
        workdir,
        f"""
        rule all:
            input: expand("out/{{i}}.txt", i=range(3))

        rule gpu:
            output: "out/{{i}}.txt"
            benchmark: "bench/{{i}}.tsv"
            resources: gpu=1
            shell: {GPU_JOB!r}
        """,
        "--resources",
        "gpu=3",
    )
    runs = [gpu_run(workdir / f"out/{i}.txt") for i in range(3)]
    last = max(range(3), key=lambda i: runs[i].start)
    first_two = [runs[i] for i in range(3) if i != last]
    assert runs[last].start >= min(earlier.end for earlier in first_two)
    assert runs[last].uuids[0] in {earlier.uuids[0] for earlier in first_two}
    assert "waits for free GPUs (1 needed)" in run.log
    benchmark = (workdir / f"bench/{last}.tsv").read_text().splitlines()
    seconds = float(benchmark[1].split("\t")[0])
    assert 1.9 < seconds < 3


def test_multi_gpu_job_skips_foreign_busy_gpu(workdir, nvidia_smi):
    nvidia_smi.reports_gpus("GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd")
    nvidia_smi.reports_busy("GPU-aaaa")
    run_snakemake(
        workdir,
        f"""
        rule all:
            input: "out/pair.txt", "out/single.txt"

        rule pair:
            output: "out/pair.txt"
            resources: gpu=2
            shell: {GPU_JOB!r}

        rule single:
            output: "out/single.txt"
            resources: gpu=1
            shell: {GPU_JOB!r}
        """,
        "--resources",
        "gpu=3",
    )
    pair = gpu_run(workdir / "out/pair.txt")
    single = gpu_run(workdir / "out/single.txt")
    assert len(pair.uuids) == 2
    assert sorted(pair.uuids + single.uuids) == ["GPU-bbbb", "GPU-cccc", "GPU-dddd"]


def test_every_kind_of_job_gets_its_gpus(workdir, nvidia_smi, monkeypatch):
    nvidia_smi.reports_gpus("GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd")
    monkeypatch.setenv("PYTHONPATH", "/driver/pythonpath")
    (workdir / "gpu.py").write_text(
        "import os, sys\n"
        "with open(snakemake.output[0], 'w') as out:\n"
        "    print(os.environ['CUDA_VISIBLE_DEVICES'], sys.executable, file=out)\n"
    )
    run = run_snakemake(
        workdir,
        """
        import os

        ENV = "echo ${{CUDA_VISIBLE_DEVICES-unset}} $PYTHONPATH > {output}"

        rule all:
            input: "plain.txt", "shell.txt", "script.txt", "run.txt", "shadow.txt"

        rule plain:
            output: "plain.txt"
            shell: ENV

        rule shell:
            output: "shell.txt"
            resources: gpu=1
            shell: ENV

        rule script:
            output: "script.txt"
            resources: gpu=1
            script: "gpu.py"

        rule run:
            output: "run.txt"
            resources: gpu=1
            run:
                with open(output[0], "w") as out:
                    print(os.environ["CUDA_VISIBLE_DEVICES"], os.getpid(), file=out)

        rule shadow:
            output: "shadow.txt"
            shadow: "minimal"
            resources: gpu=1
            shell: ENV
        """,
        "--resources",
        "gpu=4",
    )

    def words(name):
        return (workdir / name).read_text().split()

    assert words("plain.txt") == ["unset", "/driver/pythonpath"]
    shell_uuid, pythonpath = words("shell.txt")
    assert pythonpath == "/driver/pythonpath"
    script_uuid, python = words("script.txt")
    assert python == sys.executable
    run_uuid, run_pid = words("run.txt")
    assert int(run_pid) != run.pid
    shadow_uuid, _ = words("shadow.txt")
    leased = {shell_uuid, script_uuid, run_uuid, shadow_uuid}
    assert leased <= {"GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd"}
    # Spawned jobs inherit CUDA_VISIBLE_DEVICES; any call from them would be a
    # second lease by their inner snakemake.
    assert nvidia_smi.calls
    assert {visible for visible, _ in nvidia_smi.calls} == {"<unset>"}


def test_failed_gpu_jobs_release_their_gpus(workdir):
    run = run_snakemake(
        workdir,
        """
        rule all:
            input: "shell.txt", "run.txt"

        rule shell:
            output: "shell.txt"
            resources: gpu=1
            shell: "exit 3"

        rule run:
            output: "run.txt"
            resources: gpu=1
            run:
                raise ValueError("boom")
        """,
        "--resources",
        "gpu=2",
        "--keep-going",
        succeeds=False,
    )
    assert run.log.count("Released GPUs") == 2
    for uuid in ["GPU-aaaa", "GPU-bbbb"]:
        with open(workdir / "locks" / f"{uuid}.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
