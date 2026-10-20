# Changelog

## One simple notebook, random samples, three models with tuning (2026-10-17)

### Why

A review of the first version found problems that made its conclusions unreliable:

- The images were the first matching rows of the stream. The rows are not in random order, so the sample was biased.
- Midjourney was the only unseen generator in the main comparison, with 150 test images. Five more generators were tested later in a separate cell.
- Each network was trained once and the CNN was not seeded, so retraining changed the numbers. The README, `comparison_table.csv` and `common_real_comparison.csv` showed three different sets of numbers for the same networks (ResNet AUROC on SD15: 0.911, 0.680 and 0.719).
- The diffusion detector was a logistic regression on one number, so it could not show whether the reconstruction carries more information than that number.

### What changed

- The whole project is one notebook, `notebooks/Proj.ipynb`, with short commented steps from loading the data to the summary. Settings are in `config.yaml`.
- Data: one pass over the whole train split keeps a random sample of every generator (1,000 real, 1,000 SD15, 300 for each of six unseen generators). The notebook checks missing values, duplicates, original sizes and file formats before training.
- Three models: CNN, ResNet and a CNN on the residual of a pretrained diffusion model.
- Learning rate and dropout are tuned on the validation set. Every final model is trained with 3 seeds.
- Test: AUROC on SD15 and on each unseen generator, always against the same real test photos, with a chance band.
- Failure analysis: the cut-off rates of the best model and its most confident mistakes.
- Sharpness check: two detectors without deep learning and a ResNet trained with blurred copies, to test whether the models mostly follow how sharp the 32x32 pictures are.
- Training runs on the CPU by default, so the numbers repeat.
- The report is rewritten from the new results.

### Removed

Weights, results, report and reference list of the first version (`models/cnn_baseline.pt`, `models/resnet.pt`, `models/recon_classifier.joblib`, the flat files in `results/`, the old files in `report/`), and `environment.yml` (`requirements.txt` has the same packages). They stay in the git history. None of their numbers are used.

### AI assistance

This revision was written with an AI coding assistant (Claude, Anthropic). The results come from running the notebook in this repository.
