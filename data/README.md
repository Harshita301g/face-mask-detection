# Dataset: Face Mask Dataset

Kaggle: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
(about 7,500 images in two folders: `with_mask` and `without_mask`).

**Browser download (easiest):** download the zip from the link above and extract it
into this `data` folder.

**Or with the Kaggle CLI:**
```bash
kaggle datasets download -d omkargurav/face-mask-dataset -p data --unzip
```

The project finds the folder automatically as long as `with_mask` and
`without_mask` sit side by side somewhere inside `data/`, e.g.

```
data/data/with_mask/...
data/data/without_mask/...
```
If auto-detection fails, set `DATA_DIR` in `src/config.py` to the folder that
contains those two sub-folders.
