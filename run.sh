#docker run -it --rm -p 8005:8000 --mount type=bind,source="$(pwd)/data",target="/srv" tawhiri-single bash
docker run -it --rm -p 8005:8000 --mount type=bind,source="$(pwd)/data",target="/srv" tawhiri-single
