# CPU image to train or serve with mlt. For GPU training, swap the base for an
# nvidia/cuda runtime image and drop --torch-backend cpu.
#   docker build -t mlt .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" mlt mlt run configs/examples/adult.yaml
#   docker run --rm -p 8000:8000 -v "$PWD/outputs:/app/outputs" mlt mlt serve outputs/adult --host 0.0.0.0
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1
RUN pip install --no-cache-dir uv && useradd --create-home app
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
ARG EXTRAS=tune,nn,text,serve
RUN uv pip install --no-cache --torch-backend cpu ".[${EXTRAS}]"
COPY configs ./configs

RUN mkdir -p data outputs && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["mlt", "--help"]
