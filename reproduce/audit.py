"""Audit supplied PNGs and recompute archived prediction metrics, without training."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from metrics import decode_mask, confusion_matrix, metrics


def audit(repo, output):
    base = next(repo.glob('*/data'))
    output.mkdir(parents=True, exist_ok=True)
    manifest, hashes, summaries = [], defaultdict(list), {}
    historical = {policy: np.zeros((3, 3), dtype=np.int64) for policy in ('ignore', 'legacy-background')}
    historical_per_image = {policy: [] for policy in historical}
    predictions = next(repo.glob('*/J-Unet/result_pics'))
    prediction_count = 0
    for split in ('train', 'val', 'test'):
        images = {p.name: p for p in (base/split).glob('*.png')}
        labels = {p.name: p for p in (base/(split+'_labels')).glob('*.png')}
        if images.keys() != labels.keys():
            raise ValueError(f'{split}: image/label filenames do not match')
        unknown = total = 0
        class_pixels = np.zeros(3, dtype=np.int64)
        sizes = set()
        for name in sorted(images, key=lambda n: int(Path(n).stem)):
            with Image.open(images[name]) as f:
                image_array = np.array(f.convert('RGB'))
                size = f.size
            with Image.open(labels[name]) as f:
                raw_mask = np.array(f.convert('RGB'))
            if raw_mask.shape != image_array.shape:
                raise ValueError(f'{split}/{name}: shape mismatch')
            target = decode_mask(raw_mask)
            digest = hashlib.sha256(str(image_array.shape).encode()+image_array.tobytes()).hexdigest()
            hashes[digest].append(f'{split}/{name}')
            sizes.add(size)
            bad = int((target < 0).sum())
            unknown += bad
            total += target.size
            class_pixels += np.bincount(target[target >= 0], minlength=3)
            manifest.append(dict(split=split, filename=name, image_sha256=digest,
                                 mask_sha256=hashlib.sha256(raw_mask.tobytes()).hexdigest(),
                                 unknown_pixels=bad, pixels=target.size, patient_id='unknown'))
            if split == 'test' and (predictions/name).exists():
                with Image.open(predictions/name) as f:
                    pred = decode_mask(f)
                if np.any(pred < 0):
                    raise ValueError('Archived prediction has unknown colors')
                for policy in historical:
                    cm = confusion_matrix(pred, decode_mask(raw_mask, policy))
                    historical[policy] += cm
                    historical_per_image[policy].append(metrics(cm))
                prediction_count += 1
        summaries[split] = dict(images=len(images), sizes=sorted(sizes),
                                class_pixels=class_pixels.tolist(), unknown_pixels=unknown,
                                total_pixels=total, unknown_fraction=unknown/total)
    duplicates = [v for v in hashes.values() if len(v)>1]
    cross = [v for v in duplicates if len({p.split('/')[0] for p in v})>1]
    old_scores = {}
    for policy, cm in historical.items():
        old_scores[policy] = metrics(cm)
        old_scores[policy]['image_macro_averages'] = {
            key: float(np.mean([v[key] for v in historical_per_image[policy] if v[key] is not None]))
            for key in ('pixel_accuracy','dice_macro_all','miou_all','dice_macro_foreground',
                        'miou_foreground','legacy_mean_recall_background_vertebra')}
    result = dict(source_commit='b678a3130c7f2472208f33ce9c129ecc297061b0',
                  split_summary=summaries, exact_duplicate_groups=duplicates,
                  cross_split_duplicate_groups=cross,
                  patient_separation='unverifiable: patient IDs and original NIfTI absent',
                  authenticity='not independently verified; repository PNGs only',
                  archived_predictions=dict(count=prediction_count, origin='repository result_pics; weights absent; not a new model run', scores=old_scores))
    (output/'data_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (output/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); audit(a.repo,a.output)
