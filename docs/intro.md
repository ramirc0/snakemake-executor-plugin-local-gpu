This executor runs a workflow on one machine, as the `local` executor does, and gives each GPU job its own free GPUs.
A job with `resources: gpu=N` waits until N GPUs are free, then runs with `CUDA_VISIBLE_DEVICES` set to their UUIDs.
A GPU is free when no compute process uses it and no other lease holds it.
Several drivers on one node share the GPUs through `flock` locks.
Jobs without a `gpu` resource run exactly as under `local`.

It needs Snakemake with the generic software deployment API, on the `main` branch after 9.27.1.
