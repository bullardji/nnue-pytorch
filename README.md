# NNUE PyTorch

## Setup

### Docker

Use Docker with the NVIDIA PyTorch container. This eliminates the need for local Python environment setup and C++ compilation.

#### Prerequisites

For AMD Users:
- Docker
- Up-to-date ROCm driver

For NVIDIA Users:
- Docker
- Up-to-date NVIDIA driver
- NVIDIA Container Toolkit

For driver requirements, check [Running ROCm Docker containers (AMD)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) or the [PyTorch container release notes (Nvidia)](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html#rel-25-04).

The container includes CUDA 12.x / ROCm latest and all required dependencies. Your local CUDA/ROCm toolkit version doesn't matter.

### Running the container

Use the provided script to build and start the container:

```
./run_docker.sh
```

You'll be prompted to select the target GPU vendor and the path to your data directory, which will be mounted into the container. Once inside the container, you can run training commands directly.

_Building the container will take it's time and disk space (~30-60GB)_

## Network training and management

Hard way: [wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(train.py))

Easier way: [wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(easy_train.py))

## Logging

TODO: Move to wiki. Add setup for easy_train.py

```
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

## Automatically run matches to determine the best net generated by a (running) training

TODO: Move to wiki

```
python run_games.py --concurrency 16 --stockfish_exe ./stockfish.master --c_chess_exe ./c-chess-cli --ordo_exe ./ordo --book_file_name ./noob_3moves.epd run96
```

Automatically converts all `.ckpt` found under `run96` to `.nnue` and runs games to find the best net. Games are played using `c-chess-cli` and nets are ranked using `ordo`.
This script runs in a loop, and will monitor the directory for new checkpoints. Can be run in parallel with the training, if idle cores are available.


## Thanks

* Sopel - for the amazing fast sparse data loader
* connormcmonigle - https://github.com/connormcmonigle/seer-nnue, and loss function advice.
* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/
* dkappe - Suggesting ranger (https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
