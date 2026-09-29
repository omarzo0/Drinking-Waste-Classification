#!/usr/bin/env python3
"""Evaluate the trained classifier and compare Grad-CAM, LIME and SHAP on test images.

Outputs (in --out, default outputs/xai):
    xai_comparison.png   image | Grad-CAM | LIME | SHAP for each sample
    deletion_insertion.png  mean deletion / insertion curves per method
    metrics.csv          per-sample faithfulness, agreement, runtime, sanity check
    summary.csv          per-method means of the above
"""

import argparse
import os
import sys

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import xai  # noqa: E402

METHODS = ['Grad-CAM', 'LIME', 'SHAP']
ROOT = os.path.dirname(os.path.abspath(__file__))


def pick_samples(y_true, y_pred, n, rng):
    """Mostly correct predictions, plus up to a quarter misclassified ones."""
    wrong = np.flatnonzero(y_true != y_pred)
    right = np.flatnonzero(y_true == y_pred)
    n_wrong = min(len(wrong), n // 4)
    picks = list(rng.choice(wrong, n_wrong, replace=False)) if n_wrong else []
    picks += list(rng.choice(right, n - n_wrong, replace=False))
    return picks


def overlay(ax, image, heat, title):
    ax.imshow(image)
    ax.imshow(xai.to_unit(heat), cmap='jet', alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.environ.get('DATA_PATH',
                                                     os.path.join(ROOT, 'data', 'dataset_70_15_15.npz')))
    ap.add_argument('--model', default=os.path.join(ROOT, 'best_model.h5'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs', 'xai'))
    ap.add_argument('--n-samples', type=int, default=6)
    ap.add_argument('--lime-samples', type=int, default=1000, help='LIME perturbations per image')
    ap.add_argument('--shap-evals', type=int, default=1000, help='SHAP model evaluations per image')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f'Dataset not found at {args.data}. Run: python scripts/download_data.py')
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    print(f'Loading model {args.model}')
    model = tf.keras.models.load_model(args.model, compile=False)
    with np.load(args.data) as data:
        x_test, y_test = data['x_test'], data['y_test'].astype(int)

    ex = xai.Explainers(model, lime_samples=args.lime_samples, shap_evals=args.shap_evals,
                        seed=args.seed)
    print(f'Test set: {len(x_test)} images, Grad-CAM layer: {ex.layer_name}')

    y_pred = np.concatenate([ex.predict01(x_test[i:i + 256] / 255.).argmax(1)
                             for i in range(0, len(x_test), 256)])
    print(f'\nTest accuracy: {(y_pred == y_test).mean():.4f}')
    print(classification_report(y_test, y_pred, target_names=xai.CLASS_NAMES, digits=4))
    print('Confusion matrix (rows = true):')
    print(confusion_matrix(y_test, y_pred))

    samples = pick_samples(y_test, y_pred, args.n_samples, rng)
    rows, curves = [], {m: ([], []) for m in METHODS + ['Random']}
    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 3.1 * len(samples)), squeeze=False)

    for r, idx in enumerate(samples):
        image, true = x_test[idx], y_test[idx]
        pred = int(y_pred[idx])
        prob = float(ex.predict01(image[None] / 255.)[0, pred])
        print(f'\n[{r + 1}/{len(samples)}] test #{idx}: true={xai.CLASS_NAMES[true]} '
              f'pred={xai.CLASS_NAMES[pred]} ({prob:.3f})')

        maps = ex.explain(image, pred)
        maps['Random'] = (rng.random(image.shape[:2]), 0.0)
        sanity = xai.rank_correlation(maps['Grad-CAM'][0], ex.randomized_grad_cam(image, pred))

        axes[r, 0].imshow(image)
        axes[r, 0].set_title(f'#{idx} true {xai.CLASS_NAMES[true]}\npred {xai.CLASS_NAMES[pred]} '
                             f'({prob:.2f})', fontsize=9, color='black' if true == pred else 'red')
        axes[r, 0].axis('off')

        for c, method in enumerate(METHODS + ['Random']):
            heat, secs = maps[method]
            d_curve, d_auc, i_curve, i_auc = xai.deletion_insertion(ex.predict01, image, heat, pred)
            curves[method][0].append(d_curve)
            curves[method][1].append(i_curve)
            row = {'sample': idx, 'true': xai.CLASS_NAMES[true], 'pred': xai.CLASS_NAMES[pred],
                   'correct': true == pred, 'method': method, 'seconds': round(secs, 2),
                   'deletion_auc': d_auc, 'insertion_auc': i_auc}
            if method in METHODS:
                for other in METHODS:
                    if other != method:
                        row[f'spearman_vs_{other}'] = xai.rank_correlation(heat, maps[other][0])
                        row[f'top20_iou_vs_{other}'] = xai.topk_iou(heat, maps[other][0])
                overlay(axes[r, c + 1], image, heat,
                        f'{method}\ndel {d_auc:.2f} / ins {i_auc:.2f}')
            if method == 'Grad-CAM':
                row['gradcam_randomized_head_spearman'] = sanity
            rows.append(row)
            print(f'  {method:8s} {secs:6.1f}s  deletion AUC {d_auc:.3f}  insertion AUC {i_auc:.3f}')
        print(f'  Grad-CAM vs randomized-head Grad-CAM spearman: {sanity:.3f}')

    fig.suptitle('Grad-CAM vs LIME vs SHAP (evidence for the predicted class; red title = misclassified)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(os.path.join(args.out, 'xai_comparison.png'), dpi=110)
    plt.close(fig)

    fig, (ax_d, ax_i) = plt.subplots(1, 2, figsize=(11, 4))
    steps = np.linspace(0, 1, len(curves['Random'][0][0]))
    for method, (dels, ins) in curves.items():
        style = '--' if method == 'Random' else '-'
        ax_d.plot(steps, np.mean(dels, 0), style, label=method)
        ax_i.plot(steps, np.mean(ins, 0), style, label=method)
    for ax, name in ((ax_d, 'Deletion (lower AUC = more faithful)'),
                     (ax_i, 'Insertion (higher AUC = more faithful)')):
        ax.set_title(name)
        ax.set_xlabel('fraction of pixels removed' if ax is ax_d else 'fraction of pixels inserted')
        ax.set_ylabel('probability of explained class')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'deletion_insertion.png'), dpi=110)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'metrics.csv'), index=False)
    summary = df.drop(columns=['sample', 'true', 'pred', 'correct']).groupby('method', sort=False).mean()
    summary.to_csv(os.path.join(args.out, 'summary.csv'))
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.precision', 3):
        print('\nMean over samples:')
        print(summary.T)
    print(f'\nSaved results to {args.out}')


if __name__ == '__main__':
    main()
