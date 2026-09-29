#!/usr/bin/env python3
"""Numbers and figures requested by the CAISAIS 2026 reviewers, computed from the
released model and dataset.

    - images per split and class
    - test accuracy with a bootstrap 95% CI, per-class precision / recall / F1
    - confusion matrix (counts and row-normalised = per-class error rates)
    - one-vs-rest ROC curves and AUC
    - model size, parameter count and CPU inference time
    - near-duplicate check between train and test (perceptual hash)
    - error rate by image brightness (quantifies the "lighting" failure mode)

Outputs go to outputs/paper/ (report.json + figures).
"""

import json
import os
import platform
import sys
import time

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (auc, classification_report, confusion_matrix,  # noqa: E402
                             roc_auc_score, roc_curve)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ['AluCan', 'Glass', 'PET']
DATA = os.environ.get('DATA_PATH', os.path.join(ROOT, 'data', 'dataset_70_15_15.npz'))
MODEL = os.path.join(ROOT, 'best_model.h5')
OUT = os.path.join(ROOT, 'outputs', 'paper')


def dhash(images, size=8):
    """64-bit difference hash per image; robust to small shifts, scaling and compression."""
    hashes = np.empty(len(images), np.uint64)
    weights = (1 << np.arange(size * size, dtype=np.uint64))
    for i, img in enumerate(images):
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (size + 1, size),
                       interpolation=cv2.INTER_AREA)
        bits = (g[:, 1:] > g[:, :-1]).ravel().astype(np.uint64)
        hashes[i] = (bits * weights).sum()
    return hashes


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def nearest_train_distance(test_h, train_h):
    return np.array([popcount64(train_h ^ h).min() for h in test_h], int)


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    with np.load(DATA) as d:
        y = {s: d[f'y_{s}'].astype(int) for s in ('train', 'validation', 'test')}
        x_test = d['x_test']
        x_train = d['x_train']
    report['dataset'] = {
        s: {'total': int(len(v)), **{c: int((v == i).sum()) for i, c in enumerate(CLASS_NAMES)}}
        for s, v in y.items()}
    report['dataset']['all'] = {k: sum(report['dataset'][s][k] for s in y)
                                for k in ['total'] + CLASS_NAMES}
    report['dataset']['image_shape'] = list(x_test.shape[1:])
    report['dataset']['dtype'] = str(x_test.dtype)

    model = tf.keras.models.load_model(MODEL, compile=False)
    probs = np.concatenate([model.predict(x_test[i:i + 128] / 255., verbose=0)
                            for i in range(0, len(x_test), 128)])
    pred = probs.argmax(1)
    yt = y['test']
    correct = pred == yt

    rng = np.random.default_rng(0)
    boot = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(2000)]
    cm = confusion_matrix(yt, pred)
    cm_rate = cm / cm.sum(1, keepdims=True)
    cls = classification_report(yt, pred, target_names=CLASS_NAMES, output_dict=True)
    auc_ovr = {c: float(roc_auc_score(yt == i, probs[:, i])) for i, c in enumerate(CLASS_NAMES)}
    report['test'] = {
        'accuracy': float(correct.mean()),
        'accuracy_95ci_bootstrap': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        'errors': int((~correct).sum()),
        'per_class': {c: {k: float(cls[c][k]) for k in ('precision', 'recall', 'f1-score')}
                      for c in CLASS_NAMES},
        'macro_f1': float(cls['macro avg']['f1-score']),
        'confusion_matrix_counts_rows_true': cm.tolist(),
        'confusion_matrix_rate_percent_rows_true': np.round(100 * cm_rate, 2).tolist(),
        'roc_auc_ovr': auc_ovr,
        'roc_auc_macro': float(np.mean(list(auc_ovr.values()))),
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    im = ax[0].imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, f'{cm[i, j]}\n({100 * cm_rate[i, j]:.1f}%)', ha='center', va='center',
                       color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=10)
    ax[0].set_xticks(range(3), CLASS_NAMES)
    ax[0].set_yticks(range(3), CLASS_NAMES)
    ax[0].set_xlabel('Predicted class')
    ax[0].set_ylabel('True class')
    ax[0].set_title(f'Confusion matrix (test set, n = {len(yt)})')
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(yt == i, probs[:, i])
        ax[1].plot(fpr, tpr, label=f'{c} (AUC = {auc(fpr, tpr):.4f})')
    ax[1].plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax[1].set_xlabel('False positive rate')
    ax[1].set_ylabel('True positive rate')
    ax[1].set_title('One-vs-rest ROC curves (test set)')
    ax[1].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'confusion_roc.png'), dpi=300)
    plt.close(fig)

    one = x_test[:1] / 255.
    for _ in range(5):
        model.predict(one, verbose=0)
    t = []
    for _ in range(50):
        s = time.perf_counter()
        model(one, training=False)
        t.append(time.perf_counter() - s)
    s = time.perf_counter()
    model.predict(x_test[:256] / 255., batch_size=32, verbose=0)
    batch_ms = 1000 * (time.perf_counter() - s) / 256
    report['model'] = {
        'file_size_mb': round(os.path.getsize(MODEL) / 1e6, 1),
        'params_total': int(model.count_params()),
        'params_trainable': int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        'latency_ms_batch1_median': round(1000 * float(np.median(t)), 1),
        'throughput_ms_per_image_batch32': round(batch_ms, 2),
        'hardware': f'{platform.machine()} CPU, {os.cpu_count()} threads, TF {tf.__version__}',
    }

    test_h, train_h = dhash(x_test), dhash(x_train)
    dist = nearest_train_distance(test_h, train_h)
    report['near_duplicates'] = {
        'method': '64-bit dHash, Hamming distance from each test image to its nearest training image',
        **{f'test_images_within_{k}_bits': int((dist <= k).sum()) for k in (0, 2, 4, 6)},
        'fraction_within_4_bits': float((dist <= 4).mean()),
        'accuracy_on_test_without_near_duplicates(>4_bits)': float(correct[dist > 4].mean())
        if (dist > 4).any() else None,
    }
    worst = np.argsort(dist)[:6]
    nearest = [int(np.argmin(popcount64(train_h ^ test_h[i]))) for i in worst]
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
    for k, (ti, tr) in enumerate(zip(worst, nearest)):
        ax[0, k].imshow(x_test[ti])
        ax[0, k].set_title(f'test #{ti} ({CLASS_NAMES[yt[ti]]})', fontsize=8)
        ax[1, k].imshow(x_train[tr])
        ax[1, k].set_title(f'train #{tr}, dist {dist[ti]}', fontsize=8)
        ax[0, k].axis('off')
        ax[1, k].axis('off')
    fig.suptitle('Closest test/train pairs by perceptual hash (top: test, bottom: nearest train image)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'near_duplicates.png'), dpi=150)
    plt.close(fig)
    del x_train

    lum = x_test.reshape(len(x_test), -1, 3) @ np.array([0.299, 0.587, 0.114])
    brightness = lum.mean(1)
    contrast = lum.std(1)
    edges = np.percentile(brightness, [0, 25, 50, 75, 100])
    bins = np.clip(np.digitize(brightness, edges[1:-1]), 0, 3)
    report['brightness'] = {
        'quartile_edges_mean_luma_0_255': np.round(edges, 1).tolist(),
        'error_rate_percent_by_quartile(dark->bright)': [
            round(100 * float((~correct[bins == b]).mean()), 2) for b in range(4)],
        'errors_in_darkest_quartile': int((~correct[bins == 0]).sum()),
        'errors_in_brightest_quartile': int((~correct[bins == 3]).sum()),
        'mean_brightness_correct_vs_wrong': [float(brightness[correct].mean()),
                                             float(brightness[~correct].mean())],
        'mean_contrast_correct_vs_wrong': [float(contrast[correct].mean()),
                                           float(contrast[~correct].mean())],
    }

    wrong = np.flatnonzero(~correct)
    np.savetxt(os.path.join(OUT, 'misclassified_test_indices.csv'),
               np.c_[wrong, yt[wrong], pred[wrong], probs[wrong, pred[wrong]].round(4)],
               fmt=['%d', '%d', '%d', '%.4f'], delimiter=',',
               header='test_index,true,pred,confidence', comments='')
    n = min(len(wrong), 24)
    fig, ax = plt.subplots(4, 6, figsize=(15, 10.5))
    for k, a in enumerate(ax.ravel()):
        a.axis('off')
        if k < n:
            i = wrong[k]
            a.imshow(x_test[i])
            a.set_title(f'#{i} {CLASS_NAMES[yt[i]]}->{CLASS_NAMES[pred[i]]} ({probs[i, pred[i]]:.2f})',
                        fontsize=8)
    fig.suptitle('Misclassified test images (true -> predicted, confidence)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'misclassified.png'), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    json.dump(report, sys.stdout, indent=2)
    print(f'\nSaved to {OUT}')


if __name__ == '__main__':
    main()
