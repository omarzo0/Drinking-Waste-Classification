"""Explainability methods for the drinking-waste classifier and metrics to compare them.

Methods
    Grad-CAM   gradient of the class logit w.r.t. the last conv feature map (white-box)
    LIME       local linear surrogate fitted on superpixel on/off perturbations (black-box)
    SHAP       Partition explainer: Owen values over a hierarchy of image regions,
               "removed" regions are blurred (black-box)

All maps are returned at input resolution (224x224); larger = more evidence FOR the
explained class. Images are uint8 RGB in [0, 255]; the model expects x / 255.

Metrics
    deletion / insertion AUC   faithfulness: remove (or add) pixels most-important-first
                               and track the class probability (Petsiuk et al., 2018).
                               Deletion: lower is better. Insertion: higher is better.
    spearman / top-k IoU       agreement between two methods' maps
    randomization sanity check correlation between Grad-CAM on the trained model and on
                               a copy whose dense head is re-initialised
                               (Adebayo et al., 2018). High correlation = the map does
                               not depend on what the classifier learned.
"""

import time

import cv2
import numpy as np
import shap
import tensorflow as tf
from lime import lime_image
from scipy.stats import spearmanr

CLASS_NAMES = ['AluCan', 'Glass', 'PET']
IMG_SIZE = 224


def last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError('model has no 4D (convolutional) layer')


def to_unit(heat):
    """Keep positive evidence and scale to [0, 1] for display."""
    heat = np.maximum(heat, 0)
    return heat / heat.max() if heat.max() > 0 else heat


class Explainers:
    def __init__(self, model, layer_name=None, lime_samples=1000, shap_evals=1000,
                 batch_size=64, seed=0):
        self.model = model
        self.layer_name = layer_name or last_conv_layer(model)
        self.lime_samples = lime_samples
        self.shap_evals = shap_evals
        self.batch_size = batch_size
        self.seed = seed
        self.grad_model = self._grad_model(model)
        self.lime = lime_image.LimeImageExplainer(random_state=seed)
        self.shap = shap.Explainer(self.predict01,
                                   shap.maskers.Image('blur(32,32)', (IMG_SIZE, IMG_SIZE, 3)),
                                   output_names=CLASS_NAMES)

    def _grad_model(self, model):
        # Grad-CAM is defined on the pre-softmax score: softmax saturates for confident
        # predictions and would shrink every gradient towards zero.
        head = model.layers[-1]
        return tf.keras.models.Model(model.inputs,
                                     [model.get_layer(self.layer_name).output, head.input]), head

    def predict01(self, images01):
        """Class probabilities for images already scaled to [0, 1]."""
        return self.model.predict(np.asarray(images01, np.float32),
                                  batch_size=self.batch_size, verbose=0)

    def grad_cam(self, image, class_idx, grad_model=None):
        (feature_model, head) = grad_model or self.grad_model
        x = tf.convert_to_tensor(image[None] / 255., tf.float32)
        with tf.GradientTape() as tape:
            conv, penultimate = feature_model(x, training=False)
            logits = tf.matmul(penultimate, head.kernel) + head.bias
            score = logits[:, class_idx]
        grads = tape.gradient(score, conv)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    def lime_map(self, image, class_idx):
        exp = self.lime.explain_instance(image / 255., self.predict01, labels=(class_idx,),
                                         top_labels=None, hide_color=None,
                                         num_samples=self.lime_samples,
                                         batch_size=self.batch_size, random_seed=self.seed)
        heat = np.zeros(exp.segments.shape, np.float32)
        for segment, weight in exp.local_exp[class_idx]:
            heat[exp.segments == segment] = weight
        return heat

    def shap_map(self, image, class_idx):
        sv = self.shap(image[None] / 255., max_evals=self.shap_evals,
                       batch_size=self.batch_size, outputs=[class_idx], silent=True)
        return sv.values[0, ..., 0].sum(axis=-1)

    def explain(self, image, class_idx):
        """Return {method: (heatmap, seconds)}."""
        out = {}
        for name, fn in (('Grad-CAM', self.grad_cam), ('LIME', self.lime_map),
                         ('SHAP', self.shap_map)):
            t = time.perf_counter()
            heat = fn(image, class_idx)
            out[name] = (heat, time.perf_counter() - t)
        return out

    def randomized_grad_cam(self, image, class_idx):
        """Grad-CAM on a copy of the model whose dense layers are re-initialised."""
        clone = tf.keras.models.clone_model(self.model)
        clone.set_weights(self.model.get_weights())
        for layer in clone.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                init = tf.keras.initializers.GlorotUniform(seed=self.seed)
                layer.kernel.assign(init(layer.kernel.shape))
                layer.bias.assign(tf.zeros_like(layer.bias))
        return self.grad_cam(image, class_idx, grad_model=self._grad_model(clone))


def _curve(predict01, image01, heat, class_idx, steps, baseline, insertion):
    order = np.argsort(heat.ravel())[::-1]
    n = order.size
    frames = []
    for i in range(steps + 1):
        mask = np.zeros(n, bool)
        mask[order[:round(n * i / steps)]] = True
        mask = mask.reshape(heat.shape)[..., None]
        frames.append(np.where(mask, image01, baseline) if insertion
                      else np.where(mask, baseline, image01))
    probs = predict01(np.stack(frames))[:, class_idx]
    return probs, float(np.trapz(probs, dx=1 / steps))


def deletion_insertion(predict01, image, heat, class_idx, steps=20):
    """Return (deletion_curve, deletion_auc, insertion_curve, insertion_auc).

    Removed pixels are replaced by a heavily blurred copy of the image rather than
    black, which keeps the perturbed images closer to the training distribution.
    """
    image01 = image.astype(np.float32) / 255.
    baseline = cv2.blur(image01, (51, 51))
    del_curve, del_auc = _curve(predict01, image01, heat, class_idx, steps, baseline, False)
    ins_curve, ins_auc = _curve(predict01, image01, heat, class_idx, steps, baseline, True)
    return del_curve, del_auc, ins_curve, ins_auc


def _coarse(heat, size=28):
    return cv2.resize(heat.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA).ravel()


def rank_correlation(a, b):
    return float(spearmanr(_coarse(a), _coarse(b)).correlation)


def topk_iou(a, b, frac=0.2):
    a, b = _coarse(a), _coarse(b)
    k = max(1, int(frac * a.size))
    ta, tb = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
    return len(ta & tb) / len(ta | tb)
