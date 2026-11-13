"""Snakemake executor ``local-gpu``: the local executor, plus free GPUs per job.

A job with ``resources: gpu=N`` waits for N GPUs of the node that no process
uses, and runs with ``CUDA_VISIBLE_DEVICES`` set to their UUIDs. Other jobs
run as under Snakemake's ``local`` executor.
"""

import getpass
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import attr
from snakemake.exceptions import SpawnedJobError
from snakemake.executors import local
from snakemake_interface_executor_plugins.settings import ExecutorSettingsBase
from snakemake_interface_software_deployment_plugins import EnvBase

from snakemake_executor_plugin_local_gpu.gpu_pool import GpuPool

common_settings = local.common_settings


# The settings use Optional because snakemake-interface-common parses
# typing.Union types only, and `X | None` is a types.UnionType before 3.14.
@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    """Command line settings ``--local-gpu-<field>`` of the executor.

    Parameters
    ----------
    devices : str, optional
        Comma-separated GPU UUIDs or nvidia-smi indices to hand out. By
        default, the driver's ``CUDA_VISIBLE_DEVICES`` if set, else every
        GPU that nvidia-smi lists.
    poll_interval : float, optional
        Seconds between attempts while a job waits for free GPUs.
    lock_dir : Path, optional
        Directory of the GPU lock files. Drivers that share it never hand
        out the same GPU.

    Examples
    --------
    >>> ExecutorSettings(devices="0,1")  # doctest: +ELLIPSIS
    ExecutorSettings(devices='0,1', poll_interval=15.0, lock_dir=PosixPath(...))
    """

    devices: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={
            "help": "Comma-separated GPU UUIDs or nvidia-smi indices to hand "
            "out. Default: the driver's CUDA_VISIBLE_DEVICES if set, else "
            "every GPU that nvidia-smi lists.",
            "env_var": False,
            "required": False,
        },
    )
    poll_interval: Optional[float] = field(  # noqa: UP045
        default=15.0,
        metadata={
            "help": "Seconds between attempts while a job waits for free GPUs. "
            "Default: 15.",
            "env_var": False,
            "required": False,
        },
    )
    lock_dir: Optional[Path] = field(  # noqa: UP045
        default=Path(f"/tmp/snakemake-local-gpu-{getpass.getuser()}"),
        metadata={
            "help": "Directory of the GPU lock files. Drivers that share it "
            "never hand out the same GPU. Default: /tmp/snakemake-local-gpu-<user>.",
            "env_var": False,
            "required": False,
        },
    )


class Executor(local.Executor):
    """Snakemake's local executor, which leases GPUs to jobs that need them.

    Examples
    --------
    Snakemake finds the executor by its name:

    >>> from snakemake_interface_executor_plugins.registry import (
    ...     ExecutorPluginRegistry,
    ... )
    >>> ExecutorPluginRegistry().get_plugin("local-gpu").executor is Executor
    True
    """

    def __post_init__(self):
        super().__post_init__()
        settings = self.executor_settings
        self.gpus = GpuPool(
            settings.devices, settings.lock_dir, settings.poll_interval, self.logger
        )
        if "gpu" not in self.workflow.global_resources:
            self.logger.warning(
                "No --resources gpu=<number of GPUs> is set, so Snakemake starts "
                "every ready GPU job at once. Jobs that wait for a free GPU hold "
                "their cores meanwhile."
            )

    def cached_or_run(self, job, run_func, *args):
        """Run a job, on leased GPUs if it has a positive ``gpu`` resource.

        Both local paths call this in a pool thread: threaded jobs with
        ``run_wrapper`` and their run arguments, spawned jobs with
        ``spawn_job`` and the job.

        Parameters
        ----------
        job : SingleJobExecutorInterface
            The job.
        run_func : callable
            ``local.run_wrapper`` or ``self.spawn_job``.
        *args
            Arguments of ``run_func``.

        Raises
        ------
        WorkflowError
            If the job asks for more GPUs than the pool holds.

        Examples
        --------
        >>> import asyncio, logging
        >>> from types import SimpleNamespace as Namespace
        >>> workflow = Namespace(
        ...     dag=None,
        ...     main_snakefile="Snakefile",
        ...     executor_settings=ExecutorSettings(lock_dir=getfixture("tmp_path")),
        ...     execution_settings=Namespace(use_threads=False, keep_incomplete=False),
        ...     resource_settings=Namespace(cores=4),
        ...     global_resources={"gpu": 2},
        ...     workflow_settings=Namespace(cache=None),
        ...     async_run=asyncio.run,
        ... )
        >>> executor = Executor(workflow, logging.getLogger("snakemake"))

        A job without GPUs runs as under the local executor:

        >>> executor.cached_or_run(Namespace(resources={}), print, "no GPUs")
        no GPUs

        A job that asks for more GPUs than the node has fails at once:

        >>> job = Namespace(jobid=1, name="train", resources={"gpu": 3})
        >>> executor.cached_or_run(job, print, "never runs")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        snakemake_interface_common.exceptions.WorkflowError: job 1 (rule train) ...
        """
        count = job.resources.get("gpu", 0)
        if count <= 0:
            super().cached_or_run(job, run_func, *args)
            return
        with self.gpus.lease(count, f"job {job.jobid} (rule {job.name})") as uuids:
            visible_devices = ",".join(uuids)
            if run_func is local.run_wrapper:
                (run_args,) = args
                env = LeasedGpuEnv(run_args.software_env, visible_devices)
                run_args = attr.evolve(run_args, software_env=env)
                super().cached_or_run(job, local.run_wrapper, run_args)
            else:
                super().cached_or_run(job, self._spawn_job_on, job, visible_devices)

    def _spawn_job_on(self, job, visible_devices):
        cmd = self.format_job_exec(job)
        self.logger.debug(f"spawned job: {cmd}")
        env = os.environ | {"CUDA_VISIBLE_DEVICES": visible_devices}
        try:
            subprocess.check_call(cmd, shell=True, env=env)
        except subprocess.CalledProcessError as e:
            raise SpawnedJobError() from e


class LeasedGpuEnv:
    """The software environment of a job that runs on leased GPUs.

    It wraps the job's environment and exports ``CUDA_VISIBLE_DEVICES``
    inside it. All else delegates to the wrapped environment. A job
    without environment gets a stand-in that Snakemake treats as none:
    it is falsy, so ``shell`` keeps ``PYTHONPATH``, and it has no
    executables, so scripts run with Snakemake's Python.

    Parameters
    ----------
    inner : EnvBase or None
        The job's software environment.
    visible_devices : str
        Comma-separated UUIDs of the leased GPUs.

    Examples
    --------
    >>> env = LeasedGpuEnv(None, "GPU-aaaa")
    >>> bool(env), env.spec.kind, env.contains_executable("python")
    (False, None, False)
    """

    def __init__(self, inner: EnvBase | None, visible_devices: str):
        self.inner = _NoSoftwareEnv() if inner is None else inner
        self.visible_devices = visible_devices

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __bool__(self):
        return bool(self.inner)

    def managed_decorate_shellcmd(self, cmd: str) -> str:
        """Make a shell command run in the environment, on the leased GPUs.

        The export comes after the environment's activation, so the
        activation cannot override it.

        Parameters
        ----------
        cmd : str
            The shell command.

        Returns
        -------
        str
            The decorated command.

        Examples
        --------
        >>> env = LeasedGpuEnv(None, "GPU-aaaa,GPU-bbbb")
        >>> print(env.managed_decorate_shellcmd("python train.py"))
        export CUDA_VISIBLE_DEVICES=GPU-aaaa,GPU-bbbb
        python train.py
        """
        export = f"export CUDA_VISIBLE_DEVICES={self.visible_devices}"
        return self.inner.managed_decorate_shellcmd(f"{export}\n{cmd}")


class _NoSoftwareEnv:
    spec = SimpleNamespace(kind=None)

    def __bool__(self):
        return False

    def contains_executable(self, executable):
        return False

    def managed_decorate_shellcmd(self, cmd):
        return cmd
