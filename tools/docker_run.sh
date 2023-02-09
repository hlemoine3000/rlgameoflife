#!/bin/bash

set -e

# Execute in top directory.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

PROJECT_NAME=rlgameoflife

docker build -t $PROJECT_NAME .
docker run -it -u "$(id -u):$(id -g)" -v $(pwd):/$PROJECT_NAME $PROJECT_NAME bash