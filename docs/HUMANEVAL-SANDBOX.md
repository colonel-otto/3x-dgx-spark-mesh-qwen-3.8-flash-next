# HumanEval sandbox

`scripts/run_humaneval.py` requires Docker on a Linux host with the gVisor
`runsc` runtime registered. Follow the [gVisor Docker setup guide](https://gvisor.dev/docs/user_guide/quick_start/docker/).
The evaluator fails before downloading data or querying the model if its sandbox
preflight cannot execute. It never falls back to running generated code on the host.

Provision a trusted, minimal Python image before running the evaluator, for example
`docker pull python:3.11-slim`. For reproducible runs, pass the image's immutable
digest with `--sandbox-image python@sha256:...`. The evaluator uses `--pull=never`.
The image must provide Python on PATH and must not contain credentials or declare
volumes. This image and the Docker/runsc installation are trusted infrastructure.

Each completion runs as a non-root user in a separate runsc container with no
network, host mounts, host environment forwarding, or GPU passthrough. Its root
filesystem is read-only; scratch space, memory, CPU, processes, and file sizes
are limited. Candidate source arrives on stdin, and output is discarded with
Docker logging disabled to bound output storage. Timeout cleanup forcibly removes
the entire container and its descendants. Sandbox setup, inspection, and cleanup
errors abort the evaluation rather than producing a normal benchmark score.

Run local regression checks with `python -m unittest discover -s tests -p 'test_evaluation_safety.py'`.
The optional real sandbox checks require `HUMANEVAL_SANDBOX_TESTS=1` and a provisioned
Docker/runsc environment. These exercise read-only storage, blocked network access,
missing host credentials, and timeout cleanup without mounting the checkout.

The tool-calling harness separately requires
`python -m pip install -r scripts/requirements-eval.txt`.
