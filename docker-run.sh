#!/bin/bash
docker run \
    --name jax-cuda-test \
    --rm \
    -dit \
    --network=bridge \
    --user root \
    -v $(pwd):$(pwd):rw \
    -v /mnt/isabelle-data/eicl-exp/:/tmp/:rw \
    jax-cuda:latest\
    bash