FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.3 matplotlib==3.9.4 \
    && groupadd --gid 10001 sandbox \
    && useradd --uid 10001 --gid 10001 --no-create-home sandbox
COPY runner.py keeper.py collect.py /opt/sandbox/
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT []
CMD ["python", "-I", "-B", "/opt/sandbox/keeper.py"]
