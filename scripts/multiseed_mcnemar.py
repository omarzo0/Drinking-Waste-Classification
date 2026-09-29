#!/usr/bin/env python3
"""Multi-seed comparison of the six backbones with McNemar's test (Reviewers 1 and 3).

The final model keeps the ImageNet backbone frozen and trains only the head, so each
backbone's features are computed once and cached, and the head is then trained with
several seeds on the cached features. All backbones use the head and training settings
of the final DenseNet121 model:

    GAP features -> Dropout(0.405) -> Dense(128, relu) -> Dropout(0.2025) -> Dense(3, softmax)
    Adam lr 5.2e-4, batch 32, max 20 epochs, early stopping on val_loss (patience 10,
    best weights restored)

Each backbone receives its own standard ImageNet preprocessing (keras.applications
<model>.preprocess_input on 0-255 pixels), so the comparison is fair: MobileNetV3 and
EfficientNet contain their own rescaling layer and fail (chance accuracy) on [0, 1] input.
'DenseNet121-released' repeats DenseNet121 with the released model's x / 255 scaling. On-the-fly
augmentation is approximated by AUG_VIEWS pre-augmented copies of each training image
(same ImageDataGenerator settings as the notebook); each epoch draws one view per image.

McNemar's exact test compares DenseNet121 with each baseline on the test set, pairing
runs with the same seed.

Outputs: outputs/paper/multiseed_runs.csv, multiseed_summary.csv, multiseed_report.json
Features are cached in data/features/ so an interrupted run resumes.
"""

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from tensorflow.keras import applications, layers, models, optimizers, callbacks  # noqa: E402
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKBONES = ['DenseNet121', 'InceptionV3', 'ResNet50V2', 'EfficientNetB0', 'VGG19', 'MobileNetV3Small',
             'DenseNet121-released']
PREPROCESS_MODULE = {'DenseNet121': 'densenet', 'InceptionV3': 'inception_v3', 'ResNet50V2': 'resnet_v2',
                     'EfficientNetB0': 'efficientnet', 'VGG19': 'vgg19', 'MobileNetV3Small': 'mobilenet_v3'}
REFERENCE = 'DenseNet121'
AUG_VIEWS = 2


def extract(name, data, cache_dir, batch=64):
    path = os.path.join(cache_dir, f'{name}.npz')
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    arch = name.replace('-released', '')
    base = getattr(applications, arch)(input_shape=(224, 224, 3), include_top=False,
                                       weights='imagenet', pooling='avg')
    if name.endswith('-released'):
        def prep(x):
            return x / 255.
    else:
        prep = getattr(applications, PREPROCESS_MODULE[arch]).preprocess_input

    def feats(x):
        return np.concatenate([base.predict(prep(x[i:i + batch].astype(np.float32)), verbose=0)
                               for i in range(0, len(x), batch)]).astype(np.float32)

    t = time.time()
    out = {'train_0': feats(data['x_train']), 'val': feats(data['x_validation']),
           'test': feats(data['x_test'])}
    aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    for v in range(1, AUG_VIEWS + 1):
        views = []
        for i in range(0, len(data['x_train']), batch):
            xb = data['x_train'][i:i + batch].astype(np.float32)
            xb = np.stack([aug.random_transform(img, seed=hash((v, i + j)) % 2**31)
                           for j, img in enumerate(xb)])
            views.append(base.predict(prep(xb), verbose=0))
        out[f'train_{v}'] = np.concatenate(views).astype(np.float32)
    np.savez(path, **out)
    print(f'  features for {name}: dim {out["test"].shape[1]}, {time.time() - t:.0f}s')
    tf.keras.backend.clear_session()
    return out


class ViewSequence(tf.keras.utils.Sequence):
    """One randomly chosen (augmented) view per training image per epoch."""

    def __init__(self, views, y, batch, seed):
        super().__init__()
        self.views, self.y, self.batch = views, y, batch
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def on_epoch_end(self):
        n = len(self.y)
        self.order = self.rng.permutation(n)
        self.pick = self.rng.integers(0, len(self.views), n)

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch))

    def __getitem__(self, i):
        idx = self.order[i * self.batch:(i + 1) * self.batch]
        x = np.stack([self.views[self.pick[j]][j] for j in idx])
        return x, self.y[idx]


def train_head(feats, y_train, y_val, seed):
    tf.keras.utils.set_random_seed(seed)
    dim = feats['test'].shape[1]
    model = models.Sequential([
        layers.Input((dim,)),
        layers.Dropout(0.405),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.405 / 2),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizers.Adam(5.2e-4), 'categorical_crossentropy', metrics=['accuracy'])
    views = [feats[f'train_{v}'] for v in range(AUG_VIEWS + 1)]
    hist = model.fit(ViewSequence(views, tf.keras.utils.to_categorical(y_train, 3), 32, seed),
                     validation_data=(feats['val'], tf.keras.utils.to_categorical(y_val, 3)),
                     epochs=20, verbose=0,
                     callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                        restore_best_weights=True)])
    pred = model.predict(feats['test'], verbose=0).argmax(1)
    val_acc = max(hist.history['val_accuracy'])
    tf.keras.backend.clear_session()
    return pred, val_acc, len(hist.history['loss'])


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test. b = A right & B wrong, c = A wrong & B right."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--models', nargs='+', default=BACKBONES)
    args = ap.parse_args()

    cache = os.path.join(ROOT, 'data', 'features')
    out = os.path.join(ROOT, 'outputs', 'paper')
    os.makedirs(cache, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    with np.load(os.environ.get('DATA_PATH', os.path.join(ROOT, 'data', 'dataset_70_15_15.npz'))) as d:
        data = {k: d[k] for k in d.files}
    y_train, y_val, y_test = (data[k].astype(int) for k in ('y_train', 'y_validation', 'y_test'))

    rows, preds = [], {}
    for name in args.models:
        print(f'{name}: extracting features')
        feats = extract(name, data, cache)
        for seed in range(args.seeds):
            pred, val_acc, epochs = train_head(feats, y_train, y_val, seed)
            preds[(name, seed)] = pred
            rows.append({'model': name, 'seed': seed, 'epochs': epochs, 'val_acc': val_acc,
                         'test_acc': float((pred == y_test).mean()),
                         'test_macro_f1': float(f1_score(y_test, pred, average='macro'))})
            print(f'  seed {seed}: test acc {rows[-1]["test_acc"]:.4f}  (val {val_acc:.4f}, {epochs} epochs)')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'multiseed_runs.csv'), index=False)
    summary = df.groupby('model', sort=False).agg(
        test_acc_mean=('test_acc', 'mean'), test_acc_std=('test_acc', 'std'),
        test_f1_mean=('test_macro_f1', 'mean'), test_f1_std=('test_macro_f1', 'std'),
        val_acc_mean=('val_acc', 'mean'))

    report = {'protocol': __doc__.strip().split('\n\n')[1], 'seeds': args.seeds, 'mcnemar': {}}
    if REFERENCE in args.models:
        for name in args.models:
            if name == REFERENCE:
                continue
            tests = [mcnemar(y_test, preds[(REFERENCE, s)], preds[(name, s)]) for s in range(args.seeds)]
            ps = [t[2] for t in tests]
            report['mcnemar'][name] = {
                'per_seed_b_c_p': tests,
                'median_p': float(np.median(ps)), 'max_p': float(max(ps)),
                'significant_seeds_p<0.05': int(sum(p < 0.05 for p in ps))}
            summary.loc[name, 'mcnemar_median_p_vs_DenseNet121'] = float(np.median(ps))
            summary.loc[name, 'significant_seeds'] = f'{sum(p < 0.05 for p in ps)}/{args.seeds}'
    summary.to_csv(os.path.join(out, 'multiseed_summary.csv'))
    report['summary'] = json.loads(summary.to_json(orient='index'))
    with open(os.path.join(out, 'multiseed_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.round(4))
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
