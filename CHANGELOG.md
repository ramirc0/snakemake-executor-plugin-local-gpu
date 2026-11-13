# Changelog

Versions follow [Semantic Versioning](https://semver.org). The public API is
the executor name `local-gpu`, each `--local-gpu-*` setting with its default,
the lock file path `<lock_dir>/<uuid>.lock`, and the GPU UUIDs a job gets in
`CUDA_VISIBLE_DEVICES`. Drivers of different versions share the lock files, so
a change to the lock path breaks the API.

- Before 1.0.0, a breaking change is a minor release. From 1.0.0 on, it is a
  major release. Each such change goes under "Breaking".
- A new setting or behavior is a minor release.
- A bug fix that leaves the API alone is a patch release.

## 0.1.0 (unreleased)

### Added

- Executor `local-gpu`: the `local` executor plus GPU leases. A job with
  `resources: gpu=N` waits for N free GPUs, then runs with their UUIDs in
  `CUDA_VISIBLE_DEVICES`.
- A GPU is free when no compute process uses it and no `flock` lease holds it.
  Drivers that share a lock directory never hand out the same GPU.
- Settings `--local-gpu-devices`, `--local-gpu-poll-interval` and
  `--local-gpu-lock-dir`.
- Jobs that wait for GPUs stop when Snakemake cancels them.
