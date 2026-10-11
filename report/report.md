# Detecting AI-Generated Images Under Generator Shift: A CNN, a ResNet and a Diffusion Reconstruction-Error Detector

FA26 COMP 671 Deep Learning, Final Project

Author: Muthineni

## Abstract

AI image generators appear faster than detectors can be retrained, so a useful detector must work on generators it has never seen. This project tests that on Tiny-GenImage, using 32x32 images. Three detectors are trained on real photographs and fakes from Stable Diffusion 1.5 (SD15) and then tested on held-out SD15 images and on fakes from Midjourney, which none of them saw in training. The detectors are a small CNN (89,249 parameters), a small ResNet with skip connections (77,121 parameters), and a logistic regression on the reconstruction error of a pretrained diffusion model (DDPM) at noise level t=410. On SD15 the ResNet scores best (accuracy 0.793, AUROC 0.911), followed by the CNN (0.747, 0.840) and the diffusion detector (0.600, 0.614). On Midjourney all three fall to chance (accuracy 0.513, 0.477 and 0.507). The ResNet has the largest drop, so skip connections did not help on the unseen generator. The diffusion detector drops least only because it starts lowest. Midjourney fakes receive about the same average fake score as real photos, so the models treat them as ordinary photographs. SD15 fakes also had a higher, not lower, reconstruction error than real photos. The results come from one training run per model and 150 Midjourney fakes, so small differences should not be over-read, but the failure to transfer is large and consistent across all three detectors.
