# snakemake-executor-plugin-local-gpu

A Snakemake executor that runs a workflow on one machine and gives each GPU job its own free GPUs.

## Description

Snakemake's `local` executor counts a `gpu` resource but does not say which GPU a job may use.
Every job sees every GPU, so concurrent jobs pile onto device 0.
The `local-gpu` executor is the `local` executor plus GPU leases.
A job with `resources: gpu=N` waits until N GPUs are free, then runs with `CUDA_VISIBLE_DEVICES` set to their UUIDs.
A GPU is free when no compute process uses it and no other lease holds it.
Leases are `flock` locks on files in a lock directory, so several Snakemake drivers on one node share the GPUs without a daemon.
The kernel drops the locks of a driver that dies.
Jobs without a `gpu` resource run exactly as under `local`.
Rules and scripts need no changes.

The export runs inside the job's software environment, after its activation, so the environment cannot override it.
Benchmarks start after the lease, so they exclude the wait.

## Getting Started

### Dependencies

* Linux with NVIDIA GPUs and `nvidia-smi` on `PATH`.
* Python 3.11 or later.
* Snakemake with the generic software deployment API, which is on the `main` branch after 9.27.1.
  The tests pin commit `86c31130`.
  Released Snakemake 9 does not have it.

### Installing

Install the plugin into the environment that runs Snakemake:

```
pip install git+https://github.com/ramirc0/snakemake-executor-plugin-local-gpu
```

With pixi, add it under `[pypi-dependencies]`:

```toml
snakemake-executor-plugin-local-gpu = { git = "https://github.com/ramirc0/snakemake-executor-plugin-local-gpu" }
```

### Executing program

* Give each GPU rule a `gpu` resource, in the rule or in a profile's `set-resources`.
* Cap the GPUs in use with `--resources gpu=<n>`.
  Without the cap, Snakemake starts every ready GPU job at once, and jobs that wait for a GPU hold their cores.
* Run with `--executor local-gpu`.

```
snakemake --executor local-gpu --cores 64 --resources gpu=8
```

A profile does the same:

```yaml
executor: local-gpu
cores: 64
resources:
  gpu: 8
set-resources:
  train: {gpu: 1}
```

The executor reads three settings:

| Option | Default | Meaning |
| --- | --- | --- |
| `--local-gpu-devices` | `CUDA_VISIBLE_DEVICES`, else every GPU | Comma-separated GPU UUIDs or `nvidia-smi` indices to hand out |
| `--local-gpu-poll-interval` | `15` | Seconds between attempts while a job waits |
| `--local-gpu-lock-dir` | `/tmp/snakemake-local-gpu-<user>` | Lock files. Drivers that share it never hand out the same GPU |

The log shows each lease, each release and the first wait of each job.

## Help

* A job asks for more GPUs than the pool holds.
  It fails at once. Check `--local-gpu-devices` and `CUDA_VISIBLE_DEVICES`.
* A job waits although `nvidia-smi` shows idle memory.
  Any compute process makes a GPU busy, including your own jobs outside Snakemake.
  Run `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv` to see them.
* Code in a `run:` block sees `CUDA_VISIBLE_DEVICES` only when Snakemake spawns the job, which is the default.
  With `--use-threads`, only `shell()` calls inside the block see it.
* Pipe groups whose members each need GPUs can deadlock when the pool is smaller than the group.
* MIG devices never look busy, because `nvidia-smi` reports the parent GPU of each compute process.
* Users who keep separate lock directories coordinate only through `nvidia-smi`, so two jobs that start within seconds of each other can share a GPU.

```
snakemake --help | grep -A3 local-gpu
```

## Authors

Christian Ramirez (christian.ramirez1@umassmed.edu)

## Version History

See the [release history](https://github.com/ramirc0/snakemake-executor-plugin-local-gpu/releases).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

* [snakemake-executor-plugin-shared-local](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local) shares CPU and memory between local drivers with `flock` leases. It does not handle GPUs.
* [snakemake-resource-scheduler](https://github.com/mhorlacher/snakemake-resource-scheduler) assigns GPUs from inside `run:` blocks.
