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
