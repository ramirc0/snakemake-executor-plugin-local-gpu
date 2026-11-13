"""The GPUs one Snakemake driver hands out, and the locks that keep them apart."""

import fcntl
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from snakemake_interface_common.exceptions import WorkflowError

from snakemake_executor_plugin_local_gpu import nvidia_smi

UUID_PREFIXES = ("GPU-", "MIG-")


class GpuPool:
    """The GPUs that one Snakemake driver leases to its jobs.

    A lease takes GPUs that no compute process uses and that no other lease
    holds. Leases lock ``<lock_dir>/<uuid>.lock`` with ``flock``, so drivers
    that share the lock directory never hand out the same GPU. The kernel
    drops the locks of a driver that dies.

    Parameters
    ----------
    devices : str or None
        Comma-separated GPU UUIDs or nvidia-smi indices. If None, the
        driver's ``CUDA_VISIBLE_DEVICES`` if set, else every GPU that
        nvidia-smi lists.
    lock_dir : Path
        Directory of the lock files. It is created on the first lease.
    poll_interval : float
        Seconds between attempts while a lease waits for free GPUs.
    logger : logging.Logger
        Receives a message for each lease, release and first wait.

    Examples
    --------
    >>> import logging
    >>> pool = GpuPool(None, getfixture("tmp_path"), 15, logging.getLogger())
    >>> pool.uuids
    ['GPU-aaaa', 'GPU-bbbb']
    """

    def __init__(
        self,
        devices: str | None,
        lock_dir: Path,
        poll_interval: float,
        logger: logging.Logger,
    ):
        self.devices = devices
        self.lock_dir = lock_dir
        self.poll_interval = poll_interval
        self.logger = logger
        self._scan_lock = threading.Lock()

    @cached_property
    def uuids(self) -> list[str]:
        """UUIDs of the GPUs in the pool, resolved on first use.

        Raises
        ------
        WorkflowError
            If the pool is empty, names an index that nvidia-smi does not
            list, or nvidia-smi is needed and missing.

        Examples
        --------
        >>> import logging
        >>> GpuPool("1,GPU-cccc", Path("locks"), 15, logging.getLogger()).uuids
        ['GPU-bbbb', 'GPU-cccc']
        """
        devices = self.devices
        if devices is None:
            devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if devices is None:
            uuids = list(nvidia_smi.uuids_by_index().values())
        else:
            uuids = _resolve(devices.split(","))
        if not uuids:
            raise WorkflowError(
                "The local-gpu executor found no GPUs for the GPU jobs. Name "
                "them with --local-gpu-devices or CUDA_VISIBLE_DEVICES, or "
                "check that nvidia-smi lists them."
            )
        return uuids

    @contextmanager
    def lease(self, count: int, holder: str) -> Iterator[list[str]]:
        """Lease ``count`` GPUs for the duration of a ``with`` block.

        The lease takes all GPUs at once or none. It waits until enough
        GPUs are free, checking every ``poll_interval`` seconds.

        Parameters
        ----------
        count : int
            Number of GPUs.
        holder : str
            Who holds the lease, for the log messages.

        Yields
        ------
        list of str
            UUIDs of the leased GPUs.

        Raises
        ------
        WorkflowError
            If the pool holds fewer than ``count`` GPUs.

        Examples
        --------
        >>> import logging
        >>> pool = GpuPool(None, getfixture("tmp_path"), 15, logging.getLogger())
        >>> with pool.lease(1, "job 1") as first, pool.lease(1, "job 2") as second:
        ...     print(first, second)
        ['GPU-aaaa'] ['GPU-bbbb']
        """
        if count > len(self.uuids):
            raise WorkflowError(
                f"{holder} requests {count} GPUs, but the local-gpu executor "
                f"has only {len(self.uuids)}: {','.join(self.uuids)}."
            )
        locks = self._acquire(count, holder)
        uuids = list(locks)
        self.logger.info(f"Leased GPUs {','.join(uuids)} to {holder}.")
        try:
            yield uuids
        finally:
            for lock in locks.values():
                os.close(lock)
            self.logger.info(f"Released GPUs {','.join(uuids)} of {holder}.")

    def _acquire(self, count: int, holder: str) -> dict[str, int]:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        waiting = False
        while True:
            with self._scan_lock:
                locks = self._try_acquire(count)
            if locks is not None:
                return locks
            if not waiting:
                self.logger.info(f"{holder} waits for {count} free GPUs.")
                waiting = True
            time.sleep(self.poll_interval)

    def _try_acquire(self, count: int) -> dict[str, int] | None:
        busy = nvidia_smi.busy_uuids()
        locks = {}
        for uuid in self.uuids:
            if len(locks) == count:
                break
            if uuid in busy:
                continue
            lock = os.open(self.lock_dir / f"{uuid}.lock", os.O_WRONLY | os.O_CREAT)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(lock)
                continue
            locks[uuid] = lock
        if len(locks) < count:
            for lock in locks.values():
                os.close(lock)
            return None
        return locks


def _resolve(devices: list[str]) -> list[str]:
    devices = [device.strip() for device in devices if device.strip()]
    indices = [device for device in devices if not device.startswith(UUID_PREFIXES)]
    if not indices:
        return devices
    uuids_by_index = nvidia_smi.uuids_by_index()
    unknown = [index for index in indices if index not in uuids_by_index]
    if unknown:
        raise WorkflowError(
            f"nvidia-smi lists no GPU with index {','.join(unknown)}. "
            f"It lists indices {','.join(uuids_by_index)}."
        )
    return [uuids_by_index.get(device, device) for device in devices]
