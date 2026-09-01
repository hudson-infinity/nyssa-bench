# Docker

NyssaBench includes Docker assets for reproducible simulator-backed runs.

```bash
docker build -f docker/Dockerfile -t nyssa-bench:core .
docker build -f docker/Dockerfile.mujoco -t nyssa-bench:mujoco .
docker build -f docker/Dockerfile.maniskill -t nyssa-bench:maniskill .
```

Use Docker for open-source reproducibility. Kubernetes and hosted worker
orchestration belong in separate product infrastructure, not NyssaBench.
