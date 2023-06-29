# Get Started

This project is meant to be run inside the container defined in this project.

Install docker: https://docs.docker.com/desktop/install/linux-install/

Install VScode dev container.

# How it Work

```sh

# See available options
python3 -m rlgameoflife -h

# Run a simple simulation with bots to test
# The results can be found in the outputs directory
python3 -m rlgameoflife -s

# Produce a video of the latest simulation
python3 -m rlgameoflife -l

# Train an agent in the environment
python3 -m rlgameoflife -t

```

# Testing

In the dev container, you can launch each test singularly under the testing extension.
Otherwise, you can test with the command ```pytest```
