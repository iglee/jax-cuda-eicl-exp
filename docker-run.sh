docker run \
    --name jax-cuda-test \
    --rm \
    -dit \
    --network=bridge \
    --user root \
    -v $(pwd):$(pwd):rw \
    jax-cuda:latest\
    bash