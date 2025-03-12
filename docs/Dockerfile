# Docker build
# docker build --rm --tag docs:latest -f Dockerfile . --network=host
# docker run --network=host -v /mnt/stxaisvctestwe/docs4xai/dev/html:/userdocs -it --rm -p 8000:8000 docs:latest

FROM python:3.10-slim-bullseye
RUN mkdir /userdocs
WORKDIR /userdocs
WORKDIR /userdocs/docs4xai/prod/html/
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--bind", "0.0.0.0"]
