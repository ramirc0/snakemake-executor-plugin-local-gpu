import fcntl
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from snakemake_interface_common.exceptions import WorkflowError

from snakemake_executor_plugin_local_gpu.gpu_pool import GpuPool

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def lock_dir(tmp_path):
    return tmp_path / "locks"


@pytest.fixture
def new_pool(lock_dir):
    def new_pool(devices=None):
        return GpuPool(devices, lock_dir, 0.05, LOGGER)

    return new_pool


def is_locked(lock_dir, uuid):
    with open(lock_dir / f"{uuid}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def test_devices_setting_wins_over_cuda_visible_devices(new_pool, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
    assert new_pool("GPU-aaaa").uuids == ["GPU-aaaa"]


def test_cuda_visible_devices_wins_over_nvidia_smi(new_pool, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
    assert new_pool().uuids == ["GPU-bbbb"]


def test_nvidia_smi_lists_the_pool_by_default(new_pool, nvidia_smi):
    nvidia_smi.reports_gpus("GPU-aaaa", "GPU-bbbb", "GPU-cccc")
    assert new_pool().uuids == ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"]


def test_indices_map_to_uuids(new_pool, nvidia_smi, monkeypatch):
    nvidia_smi.reports_gpus("GPU-aaaa", "GPU-bbbb", "GPU-cccc")
    assert new_pool("2, 0,MIG-dddd").uuids == ["GPU-cccc", "GPU-aaaa", "MIG-dddd"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert new_pool().uuids == ["GPU-bbbb"]


def test_uuids_need_no_gpu_listing(new_pool, nvidia_smi):
    assert new_pool("GPU-xxxx,MIG-yyyy").uuids == ["GPU-xxxx", "MIG-yyyy"]
    assert nvidia_smi.calls == []


def test_pool_resolves_on_first_use(new_pool, nvidia_smi):
    pool = new_pool()
    assert nvidia_smi.calls == []
    _ = pool.uuids
    assert len(nvidia_smi.calls) == 1


def test_unknown_index_fails(new_pool):
    with pytest.raises(WorkflowError, match="no GPU with index 5"):
        _ = new_pool("0,5").uuids


def test_missing_nvidia_smi_fails(new_pool, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(WorkflowError, match="nvidia-smi is not on PATH"):
        _ = new_pool().uuids


def test_empty_pool_fails(new_pool, nvidia_smi, monkeypatch):
    nvidia_smi.reports_gpus()
    with pytest.raises(WorkflowError, match="found no GPUs"):
        _ = new_pool().uuids
    nvidia_smi.reports_gpus("GPU-aaaa")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(WorkflowError, match="found no GPUs"):
        _ = new_pool().uuids


def test_oversize_request_fails_at_once(new_pool):
    with (
        pytest.raises(WorkflowError, match="requests 3 GPUs.*has only 2"),
        new_pool().lease(3, "job 1"),
    ):
        pass


def test_concurrent_threads_get_distinct_gpus(new_pool, nvidia_smi):
    uuids = ["GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd"]
    nvidia_smi.reports_gpus(*uuids)
    pool = new_pool()
    all_hold = threading.Barrier(len(uuids))

    def lease_one(holder):
        with pool.lease(1, holder) as leased:
            all_hold.wait(timeout=5)
            return leased

    with ThreadPoolExecutor(len(uuids)) as threads:
        leases = threads.map(lease_one, [f"job {i}" for i in range(len(uuids))])
        assert sorted(uuid for leased in leases for uuid in leased) == uuids


def test_drivers_sharing_a_lock_dir_get_distinct_gpus(new_pool):
    with (
        new_pool().lease(1, "driver 1") as first,
        new_pool().lease(1, "driver 2") as second,
    ):
        assert first != second


def test_lease_takes_several_gpus(new_pool):
    with new_pool().lease(2, "job 1") as leased:
        assert leased == ["GPU-aaaa", "GPU-bbbb"]


def test_foreign_busy_gpu_is_skipped(new_pool, nvidia_smi):
    nvidia_smi.reports_busy("GPU-aaaa")
    with new_pool().lease(1, "job 1") as leased:
        assert leased == ["GPU-bbbb"]


def lease_and_release(pool, count, holder):
    with pool.lease(count, holder) as leased:
        return leased


def test_lease_waits_for_a_release(new_pool, caplog):
    pool = new_pool()
    with ThreadPoolExecutor(1) as threads, caplog.at_level(logging.INFO):
        with pool.lease(2, "job 1"):
            waiter = threads.submit(lease_and_release, pool, 1, "job 2")
            time.sleep(0.3)
            assert not waiter.done()
        assert waiter.result(timeout=5) == ["GPU-aaaa"]
    waits = [
        r
        for r in caplog.records
        if r.message == "job 2 waits for free GPUs (1 needed)."
    ]
    assert len(waits) == 1


def test_lease_waits_for_foreign_processes(new_pool, nvidia_smi):
    nvidia_smi.reports_busy("GPU-aaaa", "GPU-bbbb")
    with ThreadPoolExecutor(1) as threads:
        waiter = threads.submit(lease_and_release, new_pool(), 1, "job 1")
        time.sleep(0.3)
        assert not waiter.done()
        nvidia_smi.reports_busy("GPU-aaaa")
        assert waiter.result(timeout=5) == ["GPU-bbbb"]


def test_waiting_lease_holds_no_gpu(new_pool):
    with ThreadPoolExecutor(1) as threads:
        with new_pool().lease(1, "driver 1"):
            waiter = threads.submit(lease_and_release, new_pool(), 2, "driver 2")
            time.sleep(0.2)
            with new_pool().lease(1, "driver 3") as leased:
                assert leased == ["GPU-bbbb"]
            assert not waiter.done()
        assert waiter.result(timeout=5) == ["GPU-aaaa", "GPU-bbbb"]


def test_locks_are_released_on_exit(new_pool, lock_dir):
    with new_pool().lease(2, "job 1"):
        assert is_locked(lock_dir, "GPU-aaaa")
        assert is_locked(lock_dir, "GPU-bbbb")
    assert not is_locked(lock_dir, "GPU-aaaa")
    assert not is_locked(lock_dir, "GPU-bbbb")


def test_locks_are_released_on_exception(new_pool, lock_dir):
    with pytest.raises(RuntimeError), new_pool().lease(1, "job 1"):
        raise RuntimeError
    assert not is_locked(lock_dir, "GPU-aaaa")


def test_concurrent_first_leases_resolve_the_pool_once(new_pool, nvidia_smi):
    nvidia_smi.reports_gpus("GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd")
    pool = new_pool()
    all_start = threading.Barrier(4)

    def lease_one(holder):
        all_start.wait(timeout=5)
        return lease_and_release(pool, 1, holder)

    with ThreadPoolExecutor(4) as threads:
        list(threads.map(lease_one, [f"job {i}" for i in range(4)]))
    gpu_queries = [args for _, args in nvidia_smi.calls if "--query-gpu" in args]
    assert len(gpu_queries) == 1
