# 😷 Face Mask Detection in Real Time

Deep learning project (Machine Learning 2). A CNN (MobileNetV2 transfer learning)
classifies a face as **Mask** or **No Mask**, and OpenCV finds faces in live
webcam video so detection works in real time.

## Features
- MobileNetV2 transfer learning (train head, then fine-tune)
- Built-in data augmentation
- Evaluation: accuracy, precision/recall/F1, confusion matrix
- **Live webcam detection** with OpenCV (green box = mask, red box = no mask)
- Single-image detection and a **Streamlit web app**

## Project structure
```
face-mask-detection/
├── app/streamlit_app.py     # web demo (upload / camera photo)
├── assets/                  # bundled OpenCV face-detector file
├── data/README.md           # how to get the dataset
├── models/                  # trained model
├── results/                 # curves, confusion matrix, report
├── src/
│   ├── config.py            # paths and hyper-parameters
│   ├── data.py              # dataset loading + split
│   ├── model.py             # MobileNetV2 classifier
│   ├── train.py             # training
│   ├── evaluate.py          # test-set evaluation
│   └── detect.py            # real-time webcam / image detection
├── requirements.txt
└── LICENSE
```

## Setup
```bash
git clone https://github.com/<your-username>/face-mask-detection.git
cd face-mask-detection
python -m venv .venv
.venv\Scripts\activate           # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset
[Face Mask Dataset (Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset),
about 7,500 images. See [data/README.md](data/README.md).

## Usage
```bash
python -m src.train                       # train the model
python -m src.evaluate                    # test accuracy + confusion matrix
python -m src.detect                      # live webcam (press q to quit)
python -m src.detect --image photo.jpg    # one image -> results/annotated.jpg
streamlit run app/streamlit_app.py        # web app
```

## How it works
1. **Face detection:** OpenCV Haar cascade finds faces in each video frame.
2. **Classification:** each face crop is resized to 224x224 and passed to MobileNetV2.
3. **Output:** a coloured box and label with confidence is drawn on the frame.

## Results
Evaluated on a held-out test set of 1,120 images (never used in training):

| Metric | Value |
|--------|-------|
| Test accuracy | 99.20% |
| Precision / Recall (with_mask) | 99.0% / 99.5% |
| Precision / Recall (without_mask) | 99.5% / 98.9% |

Training curves and the confusion matrix are saved in the `results/` folder.

## Limitations
- The dataset is fairly easy: mostly clear, close-up faces. The 99% test accuracy
  may not hold on a live webcam with different lighting, angles and mask types.
- Images of the same person or similar photo style can appear in both the training
  and test splits, which can make the score slightly optimistic.
- Real-time detection also depends on the Haar cascade face detector. Side-facing,
  small or partly hidden faces may be missed before the classifier sees them. A
  stronger detector (e.g. YOLO or MediaPipe) is a good upgrade.

## License
MIT