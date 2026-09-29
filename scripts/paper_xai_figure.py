#!/usr/bin/env python3
"""Publication figure: Grad-CAM vs LIME vs SHAP for selected test images.

    python scripts/paper_xai_figure.py --indices 52 1380 664 1314

Saves outputs/paper/xai_figure.png (300 dpi, 7.16 in wide = IEEE two-column).
"""

import argparse
import os
import sys

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import xai  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indices', type=int, nargs='+', default=[52, 1380, 664, 1314])
    ap.add_argument('--data', default=os.environ.get('DATA_PATH',
                                                     os.path.join(ROOT, 'data', 'dataset_70_15_15.npz')))
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs', 'paper', 'xai_figure.png'))
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(0)
    model = tf.keras.models.load_model(os.path.join(ROOT, 'best_model.h5'), compile=False)
    with np.load(args.data) as d:
        x_test, y_test = d['x_test'], d['y_test'].astype(int)
    ex = xai.Explainers(model)

    n = len(args.indices)
    fig, axes = plt.subplots(n, 4, figsize=(7.16, 1.85 * n + 0.3), squeeze=False)
    for c, title in enumerate(['Input', 'Grad-CAM', 'LIME', 'SHAP']):
        axes[0, c].set_title(title, fontsize=9, fontweight='bold')
    for r, idx in enumerate(args.indices):
        image, true = x_test[idx], y_test[idx]
        probs = ex.predict01(image[None] / 255.)[0]
        pred = int(probs.argmax())
        maps = ex.explain(image, pred)
        tag = chr(ord('a') + r)
        axes[r, 0].imshow(image)
        axes[r, 0].set_ylabel(f'({tag}) {xai.CLASS_NAMES[true]}\n'
                              f'pred: {xai.CLASS_NAMES[pred]} ({100 * probs[pred]:.1f}%)',
                              fontsize=7.5, color='black' if true == pred else '#b00020')
        for c, method in enumerate(['Grad-CAM', 'LIME', 'SHAP']):
            heat = maps[method][0]
            _, d_auc, _, i_auc = xai.deletion_insertion(ex.predict01, image, heat, pred)
            axes[r, c + 1].imshow(image)
            axes[r, c + 1].imshow(xai.to_unit(heat), cmap='jet', alpha=0.45, vmin=0, vmax=1)
            axes[r, c + 1].set_xlabel(f'Del {d_auc:.2f} / Ins {i_auc:.2f}', fontsize=7)
        print(f'({tag}) #{idx} true={xai.CLASS_NAMES[true]} pred={xai.CLASS_NAMES[pred]} '
              f'({100 * probs[pred]:.2f}%)')
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
