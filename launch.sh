#! /bin/bash

cd /Project

hare run --rm -it -v "$(pwd)":/Project --gpus device =0 sdk36/blenderv1 /Project/run.sh

