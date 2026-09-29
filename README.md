# Drinking Waste Classification

A CNN that classifies drinking waste into **AluCan**, **Glass** and **PET**. Optuna searches over six ImageNet backbones. The best model is DenseNet121, pretrained and frozen, with a 128-unit dense head: validation accuracy 0.979 (`optuna_study_results.csv`).

This README also covers the model's explanations: Grad-CAM, compared with LIME and SHAP, plus a discussion of Grad-CAM's limitations.

## Project layout

```
classifiy.ipynb            training (Optuna), evaluation, Grad-CAM, XAI comparison
explain.py                 CLI: evaluate on the test set + compare Grad-CAM / LIME / SHAP
src/xai.py                 Grad-CAM, LIME, SHAP and comparison metrics
scripts/download_data.py   fetches the dataset from Google Drive into data/
best_model.h5              trained model (Keras 2 / TF 2.15 format)
best_waste_classifier.h5   identical copy of best_model.h5
Dockerfile, docker-compose.yml, requirements.txt
data/                      dataset (downloaded, git-ignored)
outputs/                   results (generated, git-ignored)
```

## Dataset

The preprocessed dataset is one NumPy archive, `dataset_70_15_15.npz` (about 1.6 GB). It holds 224x224 RGB images split 70 / 15 / 15:

| key | contents |
|---|---|
| `x_train`, `y_train` | training images (uint8, 0-255) and integer labels |
| `x_validation`, `y_validation` | validation split |
| `x_test`, `y_test` | held-out test split |

Labels: `0 = AluCan`, `1 = Glass`, `2 = PET`.

It is hosted on Google Drive:
https://drive.google.com/file/d/1fzN-rFtcJ9f9bdCXQIHkjQFNUkjH0ccJ/view

You don't need to download it by hand. `scripts/download_data.py` saves it to `data/dataset_70_15_15.npz` and checks that the archive is complete. It runs automatically in every Docker command and skips the download if the file is already there. To store it elsewhere, set `DATA_PATH`.

## Run with Docker (recommended)

Requires Docker with Compose v2. The image is CPU-only and works on Intel and Apple Silicon.

```bash
# 1. build the image (about 5 minutes the first time)
docker compose build

# 2. download the dataset into ./data (once; later runs reuse it)
docker compose run --rm download

# 3. evaluate the model and compare Grad-CAM, LIME and SHAP -> ./outputs/xai
docker compose run --rm explain
docker compose run --rm explain --n-samples 10 --lime-samples 2000   # options

# 4. or open the notebook in JupyterLab -> http://localhost:8888
docker compose up jupyter
```

Without Compose:

```bash
docker build -t drinking-waste-classification .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" drinking-waste-classification
```

JupyterLab runs without a password and is published on `127.0.0.1` only. Don't change the port binding to expose it on a network.

Retraining (notebook cell 4) runs 20 Optuna trials with up to 50 epochs each, and on CPU inside Docker that takes many hours. Retrain on a GPU machine (or Colab) and use the container for evaluation and explanations.

## Run locally (without Docker)

Python 3.9 - 3.11 is required. TensorFlow 2.15 does not support 3.12+.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python explain.py               # or: jupyter lab classifiy.ipynb
```

TensorFlow is pinned to **2.15** on purpose. `best_model.h5` is a Keras 2 model, and the notebook uses `layer.output_shape`, which Keras 3 (TF 2.16+) removed.

## Explainability: Grad-CAM vs LIME vs SHAP

`python explain.py` loads `best_model.h5` and prints test accuracy, a classification report and a confusion matrix. It then explains a sample of test images, including some misclassified ones, with three methods. Each method explains the **predicted** class.

| Method | Idea | Model access | Cost per image |
|---|---|---|---|
| **Grad-CAM** | Last conv feature maps (`relu`, 7x7x1024) weighted by the gradient of the class logit | gradients | one forward and one backward pass |
| **LIME** | Hides random subsets of superpixels, fits a weighted linear model to the predictions | predictions only | `--lime-samples` forward passes (default 1000) |
| **SHAP** (Partition explainer) | Owen values over a hierarchy of image regions; "absent" regions are blurred | predictions only | `--shap-evals` forward passes (default 1000) |

### How the methods are compared

| Metric | Meaning | Better |
|---|---|---|
| **Deletion AUC** | Pixels are replaced by a blurred copy in the order the method ranks them. Area under the class-probability curve. | lower |
| **Insertion AUC** | Starting from the blurred image, pixels are restored most-important-first | higher |
| **Random baseline** | Same curves for a random heatmap; a method must beat this to be informative | |
| **Spearman / top-20% IoU** | How much two methods agree, on 28x28 downsampled maps | |
| **Randomization sanity check** | Correlation between Grad-CAM on the trained model and on a copy with re-initialized dense layers (Adebayo et al., 2018) | lower |
| **seconds** | Runtime per image | lower |

Outputs in `outputs/xai/`:

- `xai_comparison.png`: each image with its Grad-CAM, LIME and SHAP overlays. The title of a misclassified image is red.
- `deletion_insertion.png`: mean deletion and insertion curves per method, including random.
- `metrics.csv` (per image) and `summary.csv` (mean per method).

The same comparison is in the last section of `classifiy.ipynb`.

## Limitations of Grad-CAM

A Grad-CAM heatmap shows **where the class score is sensitive**, not **why** the model decided. Treat it as a hypothesis to test, not as an explanation to trust.

1. **Correlated but non-causal features.** A CNN learns whatever separates the classes in the training data. That includes shortcuts which only *co-occur* with a class: background or lighting that differs between photo sessions, printed labels and logos (common on PET bottles, rare on cans), reflections typical of glass, or the typical size and position of objects in the frame. Grad-CAM highlights such a shortcut just as confidently as the object itself. A heatmap on the bottle cannot tell you whether the model responds to the material or to the label printed on it.
2. **Low resolution.** DenseNet121's last conv layer is 7x7 for a 224x224 input, so each cell covers a 32x32 patch. The upsampled map is blurry. It cannot separate the object edge from the background right behind it, or the cap from the label.
3. **Positive evidence only.** The final ReLU discards regions that push *against* the class, so you only see half of the evidence. LIME and SHAP keep both signs.
4. **First-order gradients.** Grad-CAM is a linear approximation. Saturated units have near-zero gradients even when they are important, and averaging gradients over the whole map can cancel out opposite effects. This implementation uses the pre-softmax logit, because gradients of a confident softmax output are close to zero.
5. **Depends on the chosen layer.** Earlier layers give sharper but less class-specific maps, and there is no single correct layer.
6. **Plausible is not faithful.** Some saliency methods produce nearly the same map after the model's weights are randomized (Adebayo et al., 2018). The sanity check in `explain.py` measures this for Grad-CAM on this model.
7. **One image at a time.** A handful of heatmaps inspected by eye is anecdotal evidence about one model's behavior, not validation of it.

**Testing for shortcuts:** prefer deletion/insertion scores to visual inspection. Run counterfactual tests: move the object onto a new background, cover the label, crop to the object, or change the brightness, and check whether the prediction changes. Look at where the heatmaps of misclassified images point. If Grad-CAM, LIME and SHAP all highlight the background, the model has probably learned a spurious correlation.

**LIME and SHAP are not ground truth either.** LIME depends on the superpixel segmentation and on random sampling, so results vary between runs and seeds. Both methods judge the model on perturbed images (grey or blurred patches) that it never saw in training. Both cost hundreds of times more than Grad-CAM. The three methods are most useful side by side: where they agree, the explanation is more credible, and where they disagree, look closer.

## References

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Ribeiro, Singh, Guestrin, *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*, KDD 2016 (LIME).
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017 (SHAP).
- Petsiuk, Das, Saenko, *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018 (deletion/insertion).
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
- Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence 2020.
