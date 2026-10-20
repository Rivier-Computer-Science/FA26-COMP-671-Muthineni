# Detecting AI-Generated Images Under Generator Shift

FA26 COMP 671 Deep Learning, final project, by Muthineni.

## Question

A detector is trained on real photographs and images from one generator (Stable Diffusion 1.5). Does it still work on images from generators it has never seen?

## What the notebook does

`notebooks/Proj.ipynb` is the whole project, in this order:

1. imports and settings (from `config.yaml`)
2. load the dataset (streamed from Hugging Face, 32 x 32 sample)
3. check the data: first rows, types, mean and median, missing values, duplicates, sizes and file formats
4. fill missing values and preprocess
5. train / validation / test split
6. the models and the training loop
7. diffusion residuals, the input of model 3
8. hyperparameter tuning on the validation set
9. final training with 3 seeds
10. test results on SD15 and on six unseen generators
11. plots, 12. failure analysis
13. a sharpness check: is the detector mostly following how sharp the pictures are?
14. summary

## Models

All images are 32 x 32 RGB.

| Model | Input | Course topic | Trainable parameters |
|---|---|---|---|
| CNN | pixels | chapter 10 | 89,249 |
| ResNet | pixels | chapter 11 | 77,121 |
| Diffusion CNN (the same CNN) | `abs(rebuilt - original)` from a pretrained DDPM | chapter 18 | 89,249 |

The diffusion model is `google/ddpm-cifar10-32` (trained on CIFAR-10, used as it is). Noise is added to an image, the model predicts the noise, and the image is rebuilt in one step. Images the model could have produced should be rebuilt better. The noise level `t` is chosen on validation images of real photos and SD15 only.

## Data

[Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (`TheKernel01/Tiny-GenImage`, CC BY-NC-SA 4.0) has the columns `image`, `label` (0 real, 1 fake) and `generator`. The rows are not in random order, so the notebook reads the whole train split once and keeps a random sample of every generator. The generator name is used to split the data and to report results; no model ever receives it.

| Group | Train | Validation | Test |
|---|---|---|---|
| Real | 550 | 150 | 300 |
| SD15 (the training generator) | 550 | 150 | 300 |
| Midjourney, ADM, BigGAN, GLIDE, VQDM, Wukong | none | none | 300 each |

Images are cut to a centered square and shrunk once to 32 x 32 with a bicubic filter. `data/image_table.csv` lists every chosen image and its split. Every comparison uses the same 300 real test photos.

## How the results are measured

- Metric: AUROC on the test images. 0.5 is guessing; with 300 real and 300 fake images, values between about 0.45 and 0.55 cannot be told apart from 0.5.
- Hyperparameters (learning rate and dropout) are tuned on the validation images only. The unseen generators are never used for tuning or for choosing the noise level.
- Every final model is trained with 3 seeds. The tables show the mean and the standard deviation over the seeds.
- Training: Adam, batch size 32, 30 epochs, the epoch with the lowest validation loss is kept, no augmentation. Settings are in `config.yaml` and were not changed after the test scores were computed.

## Sharpness check

Step 13 tests one explanation for the failures on unseen generators: that the models follow how sharp the 32 x 32 pictures are. It compares the ResNet with two detectors that use no deep learning (sharpness alone, and a logistic regression on brightness, contrast and sharpness) and with the same ResNet trained again on the training images plus a blurred copy of each. The test images are never blurred.

## Results

Test AUROC, mean over 3 seeds (0.5 is guessing; values between 0.454 and 0.546 cannot be told apart from guessing):

| Model | SD15 (trained on) | Wukong | Mean of Midjourney, ADM, BigGAN, GLIDE, VQDM |
|---|---|---|---|
| CNN | 0.857 | 0.886 | 0.466 |
| ResNet | 0.866 | 0.904 | 0.465 |
| Diffusion CNN | 0.733 | 0.804 | 0.466 |

- All three models detect SD15 and the related Wukong generator, and none detects the five other unseen generators (mean 0.465 to 0.466).
- The diffusion residual changes which generators are misjudged, not how many: it is above guessing on Midjourney (0.592) and far below it on BigGAN (0.390) and VQDM (0.348).
- The sharpness check (Step 13) shows that picture sharpness explains only part of this, and that training with blurred images does not help.

The full numbers, figures and discussion are in `report/report.md` (PDF: `report/report.pdf`). Every table is in `results/tables/` and every figure in `results/figures/`.

## Run it

Needs Python 3.13 or newer and an internet connection for the first run.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Open `notebooks/Proj.ipynb` with the `.venv` kernel (in VS Code, or in JupyterLab after `pip install jupyterlab`) and run all cells from top to bottom. Expect roughly 15 to 30 minutes on a laptop CPU the first time. The sampled images are saved in `data/`, so a second run skips the streaming. To sample again, delete `data/images_32x32.npy` and `data/image_table.csv`.

## Files

    notebooks/Proj.ipynb   the whole project, from the data to the summary
    config.yaml            every setting
    requirements.txt       pinned packages
    data/                  the sampled images and their table (see data/README.md)
    models/                trained weights (see models/README.md)
    results/tables/        every number as a CSV file
    results/figures/       every figure
    report/                report.md and report.pdf
    CHANGELOG.md           what changed and why

## Reproducibility

- Every random choice is seeded from `config.yaml`: the sample, the split, the network weights, the batch order, the dropout and the diffusion noise.
- Training runs on the CPU, so the numbers repeat on the same machine. Small differences are possible on another machine or another PyTorch build. The saved weights in `models/` are the ones behind every reported number.
- The sample depends on the dataset keeping its row order on Hugging Face.

## Ethics

The dataset is released for non-commercial research (CC BY-NC-SA 4.0) and the images shown are 32 x 32 versions. A detector that fails on new generators must not be used to accuse people of faking images. The report discusses this under limitations and ethics.

## AI assistance

This project was developed with an AI coding assistant. `CHANGELOG.md` lists what was changed and why. Every number in the report comes from running the notebook in this repository.
