import asyncio
import logging
from types import SimpleNamespace as Namespace

from snakemake_executor_plugin_local_gpu import Executor, ExecutorSettings


def new_executor(tmp_path, global_resources):
    workflow = Namespace(
        dag=None,
        main_snakefile="Snakefile",
        executor_settings=ExecutorSettings(lock_dir=tmp_path / "locks"),
        execution_settings=Namespace(use_threads=False, keep_incomplete=False),
        resource_settings=Namespace(cores=4),
        global_resources=global_resources,
        workflow_settings=Namespace(cache=None),
        async_run=asyncio.run,
    )
    return Executor(workflow, logging.getLogger(__name__))


def test_warns_without_a_gpu_resource_limit(tmp_path, caplog):
    new_executor(tmp_path, {"_cores": 4})
    assert "No --resources gpu=" in caplog.text


def test_is_quiet_with_a_gpu_resource_limit(tmp_path, caplog):
    new_executor(tmp_path, {"_cores": 4, "gpu": 2})
    assert caplog.text == ""


def test_job_without_gpus_never_queries_nvidia_smi(tmp_path, nvidia_smi):
    executor = new_executor(tmp_path, {"gpu": 2})
    for resources in ({}, {"gpu": 0}):
        executor.cached_or_run(Namespace(resources=resources), lambda: None)
    assert nvidia_smi.calls == []
