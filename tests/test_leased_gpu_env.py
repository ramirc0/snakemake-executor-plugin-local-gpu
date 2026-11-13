from types import SimpleNamespace

from snakemake.shell import shell

from snakemake_executor_plugin_local_gpu import LeasedGpuEnv


class FakeEnv:
    """A software env whose activation exports its own CUDA_VISIBLE_DEVICES."""

    spec = SimpleNamespace(kind="pixi")

    def managed_decorate_shellcmd(self, cmd):
        return f"export CUDA_VISIBLE_DEVICES=from-activation\n{cmd}"

    def contains_executable(self, executable):
        return executable == "python"


def run_shell(cmd, software_env):
    run_args = SimpleNamespace(
        jobid=None,
        software_env=software_env,
        threads=1,
        not_block_search_path_envvars=False,
        is_shell=True,
        bench_record=None,
        resources={},
    )
    return shell(cmd, read=True, run_args=run_args).strip()


def test_export_follows_the_activation():
    env = LeasedGpuEnv(FakeEnv(), "GPU-aaaa")
    assert env.managed_decorate_shellcmd("cmd") == (
        "export CUDA_VISIBLE_DEVICES=from-activation\n"
        "export CUDA_VISIBLE_DEVICES=GPU-aaaa\n"
        "cmd"
    )


def test_attributes_and_truth_delegate_to_the_env():
    env = LeasedGpuEnv(FakeEnv(), "GPU-aaaa")
    assert env
    assert env.spec.kind == "pixi"
    assert env.contains_executable("python")


def test_no_env_acts_as_none():
    env = LeasedGpuEnv(None, "GPU-aaaa,GPU-bbbb")
    assert not env
    assert env.spec.kind is None
    assert not env.contains_executable("python")
    assert env.managed_decorate_shellcmd("cmd") == (
        "export CUDA_VISIBLE_DEVICES=GPU-aaaa,GPU-bbbb\ncmd"
    )


def test_shell_sees_the_leased_gpus_after_activation():
    env = LeasedGpuEnv(FakeEnv(), "GPU-aaaa")
    assert run_shell("echo $CUDA_VISIBLE_DEVICES", env) == "GPU-aaaa"


def test_shell_keeps_pythonpath_without_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/driver/path")
    env = LeasedGpuEnv(None, "GPU-aaaa")
    assert run_shell("echo $PYTHONPATH $CUDA_VISIBLE_DEVICES", env) == (
        "/driver/path GPU-aaaa"
    )


def test_shell_drops_pythonpath_with_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/driver/path")
    env = LeasedGpuEnv(FakeEnv(), "GPU-aaaa")
    assert run_shell("echo ${{PYTHONPATH-unset}}", env) == "unset"
