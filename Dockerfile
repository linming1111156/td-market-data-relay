FROM python:3.13-slim-bookworm
ARG SOURCE_REVISION=unreleased
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SOURCE_REVISION=${SOURCE_REVISION}
WORKDIR /app
RUN groupadd --system --gid 10001 marketdata \
 && useradd --system --uid 10001 --gid marketdata --home-dir /nonexistent \
      --shell /usr/sbin/nologin marketdata \
 && install -d -o 10001 -g 10001 /var/lib/td-market-data/hot /var/lib/td-market-data/archive
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt
COPY collector.py export_funding_capture.py export_usdm_depth_dataset.py relay.py sync.py /app/
COPY scripts/profile_collector_hotpath.py /app/scripts/profile_collector_hotpath.py
USER 10001:10001
ENTRYPOINT ["python3", "/app/collector.py"]
