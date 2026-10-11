# Detecting AI-Generated Images Under Generator Shift: A CNN, a ResNet and a Diffusion Reconstruction-Error Detector

FA26 COMP 671 Deep Learning, Final Project

Author: Muthineni

## Abstract

AI image generators appear faster than detectors can be retrained, so a useful detector must work on generators it has never seen. This project tests that on Tiny-GenImage, using 32x32 images. Three detectors are trained on real photographs and fakes from Stable Diffusion 1.5 (SD15) and then tested on held-out SD15 images and on fakes from Midjourney, which none of them saw in training. The detectors are a small CNN (89,249 parameters), a small ResNet with skip connections (77,121 parameters), and a logistic regression on the reconstruction error of a pretrained diffusion model (DDPM) at noise level t=410. On SD15 the ResNet scores best (accuracy 0.793, AUROC 0.911), followed by the CNN (0.747, 0.840) and the diffusion detector (0.600, 0.614). On Midjourney all three fall to chance (accuracy 0.513, 0.477 and 0.507). The ResNet has the largest drop, so skip connections did not help on the unseen generator. The diffusion detector drops least only because it starts lowest. Midjourney fakes receive about the same average fake score as real photos, so the models treat them as ordinary photographs. SD15 fakes also had a higher, not lower, reconstruction error than real photos. The results come from one training run per model and 150 Midjourney fakes, so small differences should not be over-read, but the failure to transfer is large and consistent across all three detectors.

## 1. Problem Statement and Motivation

**Task.** The input is a single RGB image, resized to 32x32 pixels. The output is one label: real photograph or AI-generated. A detector is only useful if it keeps working on images from generators it was not trained on.

**Why it matters.** Realistic generated images can be used to spread false information. The authors of the GenImage benchmark point to fabricated photographs of the Pentagon on fire that circulated on social media [2]. New generators keep appearing, so a detector that only recognizes the generator it was trained on loses its value when the next one is released.

**Prior work.** Ojha et al. [1] show that a deep network trained for real-versus-fake classification on one family of generators fails on newer families, and that the real class ends up acting as a catch-all for any image that lacks the training generator's fingerprints. GenImage [2] turns this into a benchmark task, cross-generator classification, in which a detector is trained on one generator and tested on others. DIRE [3] proposes a different image representation: the error between an image and its reconstruction by a pretrained diffusion model. Its authors observe that diffusion-generated images can be approximately reconstructed while real images cannot, and report that the representation also works on diffusion models not seen in training.

**Research question.** Does the way a detector represents an image change how well it transfers to an unseen generator? This project compares two representations, raw pixels (a CNN and a ResNet) and diffusion reconstruction error, under one controlled shift: training on Stable Diffusion 1.5 (SD15) and testing on Midjourney.

**What this project is not.** It is not a reproduction of DIRE. It uses a simplified version of the idea: one noising step at a fixed noise level, a diffusion model pretrained on CIFAR-10-size images, and 32x32 inputs, so that everything runs on a laptop. The results are a small-scale test of the idea.

**Success criteria.** The primary metrics are accuracy and AUROC on two test sets: held-out SD15 images (seen generator) and Midjourney images (unseen generator). F1 and confusion matrices support them. The key quantity is the generalization gap, the seen score minus the unseen score. A detector counts as generalizing if its Midjourney AUROC is clearly above 0.5. With 150 real and 150 fake test images, an AUROC between about 0.43 and 0.57 cannot be told apart from chance (a rough 95% band from the normal approximation of the AUROC when there is no signal).

**Scope.** The project uses 1,050 images and three small models that train in minutes on a laptop, so every experiment can be repeated and inspected within a semester.

## 2. Data Description

**Source.** The data is Tiny-GenImage, hosted on Hugging Face by TheKernel01. Its dataset card lists the Kaggle dataset yangsangtai/tiny-genimage as the original source and does not say how the images were sampled. Its generator names match those of the GenImage benchmark [2]. It has 35,000 rows (28,000 in a train split, 7,000 in a validation split), takes 8.36 GB, and is licensed CC BY-NC-SA 4.0, which allows non-commercial use only. Image widths in the dataset viewer range from 34 to 4,790 pixels.

**Columns.** Each row has three fields: `image`, `label` (0 = real, 1 = fake) and `generator` (Real, ADM, BigGAN, GLIDE, Midjourney, SD14, SD15, VQDM or Wukong). The label is the prediction target. The generator column is never given to a model. It is only used to choose which images go into training and which are held back for the unseen-generator test.

**Row order.** Only the dataset's train split is used, and the project makes its own validation and test splits from it. In the first 2,000 rows, real and fake images alternate, with exactly 1,000 of each. The fakes are spread evenly over seven generators (142 or 143 images each): ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM and Wukong. SD14 did not appear in these rows. We did not check the rest of the dataset, and SD14 is not used. The metadata table had no missing values.

**Subset.** The notebook streams the train split instead of downloading the full 8.36 GB, and keeps the first 525 real, 375 SD15 and 150 Midjourney images it meets. Each image is converted to RGB, resized to 32x32 pixels without cropping (so non-square images are stretched), and scaled from 0-255 to 0-1. The real and SD15 images are shuffled with NumPy seed 42 and divided as shown in Table 1. The Midjourney images are used only for the final unseen-generator test.

| Split | Real images | Fake images | Used for |
|---|---|---|---|
| Train | 225 | 225 (SD15) | fitting the models |
| Validation | 75 | 75 (SD15) | early stopping and choosing the noise level |
| Test, seen generator | 75 | 75 (SD15) | final score on the training generator |
| Test, unseen generator | 150 | 150 (Midjourney) | final score on a new generator |

*Table 1. Data splits. No image appears in two splits, and Midjourney images appear only in the unseen-generator test. The real photographs in the unseen test are different from those used in training.*

**Integrity checks.** The notebook checks that the metadata table has no missing values, that no image contains NaN or infinite values, that every image has shape 32x32x3 with pixel values from 0 to 255 before scaling, that all 1,050 images are unique, that every split is balanced, and that no image is shared between any two of the four splits. The duplicate check compares exact pixel values after resizing, so near-duplicates would not be caught.

**Limits of the data.** The subset is small (1,050 images) and takes the first matching rows in stream order, so its object categories may not be representative; the example images we inspected were mostly animals. Resizing to 32x32 removes fine detail, and the low-level fingerprints that detectors often rely on [1] may be lost. In the dataset viewer's first rows the real, SD15 and Midjourney images are all stored as JPEG, but other aspects of their compression history were not controlled. There is only one unseen generator.
