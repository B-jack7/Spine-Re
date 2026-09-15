"""Explicit three-class metrics. Rows=truth, columns=prediction; -100=ignore."""
import numpy as np

CLASS_NAMES = ['background', 'vertebra', 'disc']


def decode_mask(rgb, policy='ignore'):
    rgb = np.asarray(rgb.convert('RGB') if hasattr(rgb, 'convert') else rgb)
    out = np.full(rgb.shape[:2], -100, dtype=np.int64)
    for cls, value in enumerate((0, 100, 255)):
        out[np.all(rgb == value, axis=-1)] = cls
    if policy == 'legacy-background':
        out[out < 0] = 0
    elif policy != 'ignore':
        raise ValueError(policy)
    return out


def confusion_matrix(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError('Prediction and target shapes differ')
    keep = target != -100
    if not np.all((target[keep] >= 0) & (target[keep] < 3)):
        raise ValueError('Invalid target class')
    if not np.all((pred[keep] >= 0) & (pred[keep] < 3)):
        raise ValueError('Invalid prediction class')
    return np.bincount(3 * target[keep].astype(int) + pred[keep].astype(int), minlength=9).reshape(3, 3)


def metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp, truth, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
    dice = divide(2 * tp, truth + pred)
    iou = divide(tp, truth + pred - tp)
    recall = tp / (truth + 1e-10)
    def mean(x):
        return float(np.nanmean(x)) if np.any(np.isfinite(x)) else None
    def safe(x):
        return [float(v) if np.isfinite(v) else None for v in x]
    return dict(pixel_accuracy=float(tp.sum()/cm.sum()) if cm.sum() else None,
                dice_per_class=safe(dice), iou_per_class=safe(iou),
                dice_macro_all=mean(dice), dice_macro_foreground=mean(dice[1:]),
                miou_all=mean(iou), miou_foreground=mean(iou[1:]),
                # Historical trainer called this "accuracy", and excluded disc.
                legacy_mean_recall_background_vertebra=mean(recall[:2]),
                valid_pixels=int(cm.sum()), confusion=cm.astype(int).tolist())
