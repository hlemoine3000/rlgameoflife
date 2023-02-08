#!/bin/bash

# Execute in top directory.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

docker run --rm -v $(pwd):/data cytopia/black rlgameoflife/
