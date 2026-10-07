# LAB's sandbox image with sandboxd as its entrypoint, for Agent Sandbox.
# BASE is LAB's image built from its sandbox/ directory; cloudbuild.yaml
# builds both at the pinned LAB revision.
ARG BASE
FROM registry.k8s.io/agent-sandbox/sandboxd:v1.0.5 AS sandboxd

FROM ${BASE}
COPY --from=sandboxd /usr/local/bin/sandboxd /usr/local/bin/sandboxd
RUN chown -R 1000:1000 /workspace
# LibreOffice and pip want a writable home.
ENV HOME=/tmp
USER 1000
# Root / keeps LAB's absolute paths; the pod is the isolation boundary.
ENTRYPOINT ["/usr/local/bin/sandboxd", "--root-dir=/"]
