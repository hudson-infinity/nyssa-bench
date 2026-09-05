ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG PYTHON_VERSION
ARG NYSSA_VERSION
ARG NYSSA_COMMIT
ARG NYSSA_BUILD_DATE
ARG TARGETPLATFORM

LABEL org.opencontainers.image.title="NyssaBench core" \
      org.opencontainers.image.description="Core NyssaBench CLI and protocol runtime" \
      org.opencontainers.image.source="https://github.com/hudson-infinity/nyssa-bench" \
      org.opencontainers.image.version="${NYSSA_VERSION}" \
      org.opencontainers.image.revision="${NYSSA_COMMIT}" \
      org.opencontainers.image.created="${NYSSA_BUILD_DATE}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.hudson-labs.nyssa.python="${PYTHON_VERSION}" \
      io.hudson-labs.nyssa.simulator="none"

ENV NYSSA_CONTAINER_PROFILE=core \
    NYSSA_CONTAINER_VERSION=${NYSSA_VERSION} \
    NYSSA_CONTAINER_COMMIT=${NYSSA_COMMIT} \
    NYSSA_CONTAINER_BUILD_DATE=${NYSSA_BUILD_DATE} \
    NYSSA_CONTAINER_PLATFORM=${TARGETPLATFORM} \
    NYSSA_CONTAINER_PYTHON=${PYTHON_VERSION} \
    NYSSA_CONTAINER_SIMULATOR=none \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY dist/*.whl /tmp/nyssa-dist/
RUN wheel="$(find /tmp/nyssa-dist -maxdepth 1 -name '*.whl' -print -quit)" \
    && test -n "${wheel}" \
    && python -m pip install --no-cache-dir "${wheel}" \
    && python -m pip check \
    && rm -rf /tmp/nyssa-dist \
    && useradd --create-home --uid 10001 nyssa \
    && install -d -o nyssa -g nyssa /workspace

USER nyssa
WORKDIR /workspace

CMD ["nyssa", "list-suites"]
