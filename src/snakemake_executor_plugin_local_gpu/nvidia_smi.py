"""The two nvidia-smi queries that GPU leasing needs."""

import subprocess

from snakemake_interface_common.exceptions import WorkflowError


def uuids_by_index() -> dict[str, str]:
    """Map the index of each GPU that nvidia-smi lists to its UUID.

    nvidia-smi numbers GPUs in PCI bus order. CUDA numbers them fastest
    first unless ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, so only the UUID names
    the same GPU for both.

    Returns
    -------
    dict of str to str
        GPU UUID keyed by nvidia-smi index, both as strings.

    Raises
    ------
    WorkflowError
        If nvidia-smi is missing or fails.

    Examples
    --------
    On a node with two GPUs:

    >>> uuids_by_index()
    {'0': 'GPU-aaaa', '1': 'GPU-bbbb'}
    """
    rows = (line.split(",") for line in _query("--query-gpu=index,uuid"))
    return {index.strip(): uuid.strip() for index, uuid in rows}


def busy_uuids() -> set[str]:
    """Return the UUIDs of the GPUs that run at least one compute process.

    nvidia-smi lists the processes of every user, so this also sees GPU
    jobs that run outside Snakemake.

    Returns
    -------
    set of str
        UUIDs of the busy GPUs.

    Raises
    ------
    WorkflowError
        If nvidia-smi is missing or fails.

    Examples
    --------
    On a node where no process uses a GPU:

    >>> busy_uuids()
    set()
    """
    return set(_query("--query-compute-apps=gpu_uuid"))


def _query(query: str) -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise WorkflowError(
            "nvidia-smi is not on PATH. The local-gpu executor needs it to "
            "find GPUs and to see which ones are busy."
        ) from e
    except subprocess.CalledProcessError as e:
        raise WorkflowError(
            f"nvidia-smi {query} failed with exit code {e.returncode}: "
            f"{e.stderr.strip()}"
        ) from e
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
