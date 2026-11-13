### Usage

Give each GPU rule a `gpu` resource, cap the GPUs in use, and select the executor:

```
snakemake --executor local-gpu --cores 64 --resources gpu=8 --set-resources train:gpu=1
```

Without `--resources gpu=<n>`, Snakemake starts every ready GPU job at once, and jobs that wait for a GPU hold their cores.

The export of `CUDA_VISIBLE_DEVICES` runs inside the job's software environment, after its activation.
Benchmarks start after the lease, so they exclude the wait.

### Limits

* Code in a `run:` block sees `CUDA_VISIBLE_DEVICES` only when Snakemake spawns the job, the default. With `--use-threads`, only `shell()` calls inside the block see it.
* Pipe groups whose members each need GPUs can deadlock when the pool is smaller than the group.
* MIG devices never look busy, because `nvidia-smi` reports the parent GPU of each compute process.
