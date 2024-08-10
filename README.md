**Built on top of [gorodnitskiy/jax-cuda-docker](https://github.com/gorodnitskiy/jax-cuda-docker) 🫡**

my 2 🪙's: jax-cuda-tf compatibility is a nightmare. instead of trying to reconfigure your servers, it is much easier to work with containerized environments like docker.

# JAX with CUDA support in Docker

There are a lot of issues on GitHub about installing JAX with CUDA support, related to JAX and CUDA/cuDNN versions
mismatching. This repository contains `Dockerfile` that can be used to easily run JAX with CUDA support in Docker, though specific modifications may be necessary in places.

***For example, for eicl experiments***, you need a very specific version of cudnn+jax combination. After many, MANY trials and error, use
```
pip install "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
inside the docker container (i.e. `docker exec -it` into it). While the [Dockerfile](https://github.com/iglee/jax-cuda-eicl-exp-docker/blob/master/Dockerfile) 
automatically finds jax-cudnn combo, you may sometimes need to be über specific in case of the pesky lil updates 
that break everything. Otherwise, the script defaults to the most up-to-date `jax`/`jaxlib` available for said `cuda`/`cudnn`.

## Build

It strictly requires to specify, based on existing nvidia docker images on
[NVIDIA Docker hub](https://hub.docker.com/r/nvidia/cuda/tags):

- CUDA (eg: `11.4.3`)
- OS (eg: `ubuntu22.04` or `centos7`)

In case of JAX and CUDA/CUDNN versions mismatching, you have to change `CUDA` and `JAX_CUDA_CUDNN` building variables.

Check JAX versions via [Google Storage](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).
Check CUDA/cuDNN versions matching via [cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive).

Each JAX for CUDA compiled with specific cuDNN versions. For example `jaxlib==0.4.2` (CUDA=11) compiled for two
cuDNN versions: 8.2 or 8.6. So, we might choose:

- `CUDA`="11.4.3" and `JAX_CUDA_CUDNN`="cuda11_cudnn82"
- `CUDA`="11.8.0" and `JAX_CUDA_CUDNN`="cuda11_cudnn86"

Also, it might be a problem with overall NVIDIA environment, for example incompatible NVIDIA driver version for
requested CUDA version. It has to be checked apart.

### Additionally, I highly recommend configuring conda environment as part of the docker build.
An example is shown [here](https://github.com/iglee/jax-cuda-eicl-exp-docker/blob/master/environment.yaml). You can also specify
pip requirements like in the example.

### Putting this all together...
For example docker builds, take a look at [this snippet](https://github.com/iglee/jax-cuda-eicl-exp/blob/master/docker-build.sh).

## Run

See [example here](https://github.com/iglee/jax-cuda-eicl-exp/blob/master/docker-run.sh).
