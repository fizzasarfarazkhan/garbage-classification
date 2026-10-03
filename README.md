# ♻️ Garbage Classification

An image classification project that predicts the type of waste from a photo into one of 6 classes: **cardboard, glass, metal, paper, plastic, trash**. Three approaches are compared: a classical ML baseline, a CNN built from scratch, and transfer learning with MobileNetV2.

**🔗 Live demo:** (https://garbage-classification-fizza.streamlit.app)
> If the app is asleep, click the wake-up button and wait about 30 seconds.

## Results

Evaluated on a held-out test set of 503 images (20% split).

| Model | Test accuracy |
|---|---|
| Logistic Regression (raw pixels, baseline) | 35.4% |
| CNN (from scratch) | 51.5% |
| **MobileNetV2 (transfer learning)** | **74.2%** |

### MobileNetV2 per-class performance

| Class | Precision | Recall | F1-score | Test images |
|---|---|---|---|---|
| cardboard | 0.95 | 0.65 | 0.77 | 80 |
| glass | 0.75 | 0.79 | 0.77 | 100 |
| metal | 0.77 | 0.72 | 0.74 | 82 |
| paper | 0.70 | 0.92 | 0.80 | 118 |
| plastic | 0.74 | 0.66 | 0.70 | 96 |
| trash | 0.46 | 0.41 | 0.43 | 27 |

Macro F1: 0.70, weighted F1: 0.74.

## Key observations

- **Classical ML is not enough for images.** Logistic Regression on flattened pixels reached only 35.4% (random guessing is about 16.7%).
- **The CNN trained from scratch overfit.** Training accuracy reached about 98% while test accuracy stayed near 50%, meaning the model memorised the training images.
- **Transfer learning fixed this.** Using a pretrained MobileNetV2 as a frozen feature extractor raised test accuracy to 74.2% with only 5 epochs of training.
- **Limitation:** the "trash" class has far fewer images (27 in the test set vs 80-118 for other classes), so its F1-score is the lowest. Future work: data augmentation, class weights and fine-tuning the MobileNetV2 layers.

## Tech stack

Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn, Streamlit.

## Project structure

```
garbage-classification/
├── Garbage_Classification.ipynb   # training and evaluation notebook
├── garbage_model.keras            # trained MobileNetV2 model
├── streamlit_app.py               # web app
├── requirements.txt
├── samples/                       # sample images to test the app
└── Garbage_Classification_Final_Report.pdf
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dataset

(https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) - 6 classes, split 80% training / 20% testing, images resized to 128x128 and normalised to [0, 1].

## Author

Fizza khan - (https://www.linkedin.com/in/fizzasarfarazkhan) - fizzakh2003@gmail.com
