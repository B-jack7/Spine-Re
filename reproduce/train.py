"""Fresh, traceable experiments on the supplied PNGs. See README.md for limits."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from metrics import decode_mask, confusion_matrix, metrics


class PngDataset(Dataset):
    def __init__(self, root, split, policy, limit=0):
        self.root, self.split, self.policy = root, split, policy
        self.files = sorted((root/split).glob('*.png'), key=lambda p: int(p.stem))
        labels = {p.name for p in (root/(split+'_labels')).glob('*.png')}
        if not self.files or {p.name for p in self.files} != labels:
            raise ValueError('Missing data or mismatched image/label filenames')
        if limit:
            self.files = [self.files[i] for i in np.linspace(0, len(self.files)-1, min(limit,len(self.files)), dtype=int)]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]
        with Image.open(p) as f:
            arr = np.array(f.convert('RGB'), dtype=np.float32)/255
        with Image.open(self.root/(self.split+'_labels')/p.name) as f:
            mask = decode_mask(f, self.policy)
        if arr.shape != (512,512,3) or mask.shape != (512,512):
            raise ValueError('Native 512x512 inputs required; masks are never resized')
        if not np.any(mask >= 0):
            raise ValueError('Mask has no valid labels')
        arr = (arr-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        return torch.from_numpy(arr.transpose(2,0,1).copy()), torch.from_numpy(mask), p.name


def make_model(repo, name):
    base = next(repo.glob('*/J-Unet'))
    sys.path.insert(0,str(base))
    if name == 'junet':
        from nets.unet3plus import UNet_3Plus
        return UNet_3Plus(3)
    sys.path.insert(0,str(base.parent/'UNET'))
    from unet.unet_model import UNet
    return UNet(3,3)


def dice_loss(logits, target):
    valid = target != -100
    onehot = nn.functional.one_hot(target.clamp_min(0),3).permute(0,3,1,2).float()
    prob = logits.float().softmax(1)
    mask = valid.unsqueeze(1)
    prob, onehot = prob*mask, onehot*mask
    score = (2*(prob*onehot).sum((2,3))+1)/((prob+onehot).sum((2,3))+1)
    return 1-score.mean()


def atomic_save(obj,path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)


def evaluate(model, loader, device, amp, output=None):
    model.eval()
    total = np.zeros((3,3),dtype=np.int64)
    rows = []
    with torch.inference_mode():
        for x,y,names in loader:
            with torch.autocast(device_type=device.type,enabled=amp):
                # Original baseline emits tensor sizes from forward().
                with contextlib.redirect_stdout(io.StringIO()):
                    pred = model(x.to(device)).argmax(1).cpu().numpy()
            for j,name in enumerate(names):
                cm = confusion_matrix(pred[j],y[j].numpy())
                total += cm
                rows.append(dict(filename=name,**metrics(cm)))
                if output:
                    Image.fromarray(np.array([0,100,255],dtype=np.uint8)[pred[j]]).save(output/name)
    result = metrics(total)
    result['images'] = len(rows)
    result['image_macro_averages'] = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
        for k in ('dice_macro_all','dice_macro_foreground','miou_all','miou_foreground','pixel_accuracy')}
    return result, rows


def run(a):
    repo = Path(a.repo).resolve(); output = Path(a.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() and not a.resume:
        raise ValueError('Output already contains an experiment: choose a new directory or --resume')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not a.allow_cpu:
        raise RuntimeError('A Colab GPU is required. CPU training needs explicit --allow-cpu.')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = next(repo.glob('*/data'))
    train = PngDataset(root,'train',a.label_policy,a.train_limit)
    val = PngDataset(root,'val',a.label_policy,a.val_limit)
    test = PngDataset(root,'test',a.label_policy,a.test_limit)
    gen = torch.Generator().manual_seed(a.seed)
    tr_loader = DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0,generator=gen)
    va_loader = DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
    te_loader = DataLoader(test,batch_size=1,shuffle=False,num_workers=0)
    model = make_model(repo,a.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=a.lr)
    amp = device.type == 'cuda' and not a.no_amp
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    config = dict(vars(a),python=sys.version,torch=torch.__version__,numpy=np.__version__,
                  device=str(device),gpu=torch.cuda.get_device_name(0) if amp else None,
                  parameters=sum(p.numel() for p in model.parameters()),
                  counts={'train':len(train),'val':len(val),'test':len(test)},
                  aggregation='global confusion + image macro; all classes and foreground separately',
                  limitations=['patient IDs unavailable','PNG source not independently verified',
                               'not exact paper reproduction','AMP changes numerical precision'],
                  git_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
                  source_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in list((repo/'reproduce').glob('*.py'))+list(next(repo.glob('*/J-Unet/nets')).glob('*.py'))})
    start, best = 0, -1.0
    if a.resume:
        ckpt = torch.load(output/'last.pt',map_location=device,weights_only=False)
        prior = json.loads((output/'config.json').read_text())
        for key in ('model','seed','label_policy','batch_size','lr','train_limit','val_limit','test_limit','no_amp','loss'):
            if prior[key] != config[key]: raise ValueError(f'Resume configuration changed: {key}')
        if prior['source_hashes'] != config['source_hashes']:
            raise ValueError('Source changed since checkpoint; start a new experiment')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler']); gen.set_state(ckpt['loader_rng'].cpu())
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([v.cpu() for v in ckpt['cuda_rng']])
        start,best = ckpt['epoch'],ckpt['best']
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config),flush=True)
    started=time.monotonic()
    for epoch in range(start,a.epochs):
        model.train()  # Restore BatchNorm training mode after every validation.
        losses=[]
        for step,(x,y,_) in enumerate(tr_loader):
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                with contextlib.redirect_stdout(io.StringIO()): logits=model(x)
                loss = dice_loss(logits,y) if a.loss=='dice' else nn.functional.cross_entropy(logits,y,ignore_index=-100)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no fabricated result or silent restart')
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            losses.append(loss.item())
            if step%25==0: print(f'epoch={epoch+1} step={step+1}/{len(tr_loader)} loss={loss.item():.6f}',flush=True)
        result,_ = evaluate(model,va_loader,device,amp)
        score=result['dice_macro_foreground']
        improved = score is not None and score>best
        if improved:
            best=score
            atomic_save(dict(model=model.state_dict(),epoch=epoch+1),output/'best.pt')
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                         epoch=epoch+1,best=best,loader_rng=gen.get_state(),torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),output/'last.pt')
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation=result,elapsed_seconds=time.monotonic()-started)
        with (output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    best_ckpt=torch.load(output/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(best_ckpt['model'])
    pred_dir=output/'predictions';pred_dir.mkdir(exist_ok=True)
    result,rows=evaluate(model,te_loader,device,amp,pred_dir)
    result.update(best_epoch=best_ckpt['epoch'],label_policy=a.label_policy,
                  experiment='new_training_on_repository_PNGs',
                  full_dataset=not any((a.train_limit,a.val_limit,a.test_limit)),
                  planned_epochs=a.epochs,source_checkpoint='best.pt',
                  patient_independent_test='unverified')
    (output/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'test_per_image.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('TEST_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['junet','unet'],default='junet')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--loss',choices=['dice','ce'],default='dice')
    p.add_argument('--label-policy',choices=['ignore','legacy-background'],default='ignore')
    for split in ('train','val','test'):p.add_argument(f'--{split}-limit',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--no-amp',action='store_true')
    p.add_argument('--allow-cpu',action='store_true')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or min(a.train_limit,a.val_limit,a.test_limit)<0:p.error('Invalid count')
    run(a)
