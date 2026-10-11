# Detecting AI-Generated Images Under Generator Shift

Course project for FA26 COMP 671 (Deep Learning) by Muthineni.

## Question

Does a detector trained on real photos and fakes from one image generator (Stable Diffusion 1.5) still work on fakes from a generator it has never seen (Midjourney)?

## Approach

Three detectors are trained and tested on the same data. All images are 32x32.

| Detector | Course topic | Description |
|---|---|---|
| CNN | Chapter 10 | 3 convolution layers, dropout 0.3, 89,249 parameters |
| ResNet | Chapter 11 | 3 residual blocks with skip connections, dropout 0.3, 77,121 parameters |
| Diffusion reconstruction error | Chapter 18 | A pretrained DDPM (`google/ddpm-cifar10-32`) adds noise to each image at t=410, predicts the noise and rebuilds the image. The squared error between the original and the rebuilt image is one number per picture, and a logistic regression classifies it. |

The noise level t=410 was chosen on validation data (real vs SD15 only), never by looking at Midjourney. The pretrained diffusion model was trained on CIFAR-10, not on this dataset. Training a diffusion model from scratch was not feasible on a laptop.

## Data

Tiny-GenImage (`TheKernel01/Tiny-GenImage` on Hugging Face) has three columns: `image`, `label` (0 = real, 1 = fake) and `generator` (Real, ADM, BigGAN, GLIDE, Midjourney, SD14, SD15, VQDM, Wukong). The full dataset is about 8 GB, so the notebook streams it and keeps only the images it needs.

The notebook collects the first matching images in stream order: 525 real, 375 SD15 and 150 Midjourney. Pixels are scaled to 0-1. The real and SD15 images are shuffled with numpy seed 42 and split as follows.

| Split | Real | Fake | Used for |
|---|---|---|---|
| Train | 225 | 225 (SD15) | training |
| Validation | 75 | 75 (SD15) | early stopping and choosing t |
| Test, seen generator | 75 | 75 (SD15) | final score |
| Test, unseen generator | 150 | 150 (Midjourney) | final score |

Checks in the notebook: no missing values, no NaN or infinite pixels, no duplicate images, balanced classes in every split, and no image shared between two splits.

## Results

| Model | SD15 accuracy | Midjourney accuracy | SD15 AUROC | Midjourney AUROC |
|---|---|---|---|---|
| CNN | 0.747 | 0.513 | 0.840 | 0.492 |
| ResNet | 0.793 | 0.477 | 0.911 | 0.458 |
| Diffusion error | 0.600 | 0.507 | 0.614 | 0.527 |

- All three detectors work on SD15 and fall to chance on Midjourney.
- The ResNet is best on SD15 and has the biggest drop, so the residual architecture did not help on the unseen generator.
- The diffusion detector has the smallest drop only because it starts lower. It did not generalize better.
- Midjourney fakes get about the same average "fake" score as real photos (0.26-0.27 vs 0.25-0.29 for the CNN and ResNet), so nothing in the scores separates them from real photos.

Limits: 150 Midjourney fakes, one unseen generator, 32x32 images, and each model trained once. At this sample size an AUROC between about 0.43 and 0.57 cannot be told apart from 0.5.

## Repository layout

    notebooks/Proj.ipynb    the whole project, with outputs saved
    src/check_env.py        prints package versions and the device in use
    models/                 cnn_baseline.pt, resnet.pt, recon_classifier.joblib
    results/                metrics (json), tables (csv) and figures (png)
    config.yaml             seeds, splits, training settings
    requirements.txt        pinned packages (tested route)
    environment.yml         the same packages in conda format (not tested)
    report/                 final written report
    data/                   empty, the data is streamed

## Setup and run

Tested on a Mac with Python 3.14.7.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python src/check_env.py

Open `notebooks/Proj.ipynb` in VS Code, choose the `.venv` kernel and use Run All. The notebook must run from the `notebooks/` folder because it saves to `../results` and `../models` (VS Code does this by default). It uses the Apple GPU (mps) if available and the CPU otherwise.

The image-collection cell streams about 6,000 rows over the internet and takes several minutes. The first run also downloads the 150 MB diffusion model.

## Reproducibility notes

- The data split uses seed 42. `torch.manual_seed(42)` was set before ResNet training, after the model was created, so it fixes the batch order but not the starting weights.
- The CNN was not seeded at all. Re-training either network therefore gives slightly different numbers. The trained weights are saved in `models/` and the reported results come from them.
- GPU training is not bit-for-bit repeatable, so small differences are possible for every model.
- The subset depends on the dataset keeping the same row order on Hugging Face.
