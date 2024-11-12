#!/usr/bin/env bash
sudo docker build -t lab3/image:0.0.1 .
sudo docker run -it --rm -p 8888:8888 --gpus all -v "${PWD}":/app lab3/image:0.0.1