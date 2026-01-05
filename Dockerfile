FROM sourcepole/qwc-uwsgi-base:alpine-v2025.10.13

WORKDIR /srv/qwc_service
ADD pyproject.toml uv.lock ./

RUN \
  apk add --no-cache --update --virtual runtime-deps gdal && \
  apk add --no-cache --update --virtual build-deps g++ python3-dev gdal-dev musl-dev && \
  uv sync --frozen && \
  uv cache clean && \
  apk del build-deps

ADD src /srv/qwc_service/

ENV SERVICE_MOUNTPOINT=/api/v1/oblique-imagery
