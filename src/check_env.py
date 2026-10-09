# check_env.py
# Prints which libraries are installed and picks the device (cuda / mps / cpu)
# that the rest of the project will run on.

import sys
import torch


def get_device(force_cpu=False):
    # Prefer an NVIDIA GPU, then the Apple GPU, then plain CPU.
    # force_cpu is there in case MPS acts weird later on.
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# other scripts will import this one value so we never hardcode "cuda"
DEVICE = get_device()


if __name__ == "__main__":
    import torchvision
    import datasets
    import diffusers
    import sklearn
    import pandas
    import matplotlib

    print("Python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("datasets:", datasets.__version__)
    print("diffusers:", diffusers.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("pandas:", pandas.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("Using device:", DEVICE)