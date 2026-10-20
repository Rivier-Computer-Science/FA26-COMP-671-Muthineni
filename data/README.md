# data/

Written by step 2 of `notebooks/Proj.ipynb`.

| File | What it is |
|---|---|
| `images_32x32.npy` | the sampled images, `uint8`, shape (N, 32, 32, 3), in the order of `image_table.csv` |
| `image_table.csv` | one row per image: `generator`, `label` (0 real, 1 fake), `native_w`, `native_h`, `format` (size and file format of the original image), `row_number` (row in the Hugging Face split) and `split` (train, val or test) |

Source: [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) on Hugging Face, licensed CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). The images here are 32 x 32 versions of images from that dataset, kept for coursework reproducibility under the same license.
