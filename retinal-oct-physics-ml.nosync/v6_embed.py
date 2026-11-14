"""v6 / Step 3a: frozen deep embeddings of the flattened-retina tile, cached once per scan.

  ResNet-18 (ImageNet, torchvision)        -> 512-d global-average-pooled features
  DINOv2-S  (timm vit_small_patch14_dinov2) -> 384-d CLS token

Input: the 128 x 512 flattened tile (masked pixels = 0), replicated to 3 channels, ImageNet
normalisation; DINOv2 gets 126 x 504 (multiple of its 14-px patch). No fine-tuning: the networks
are fixed feature extractors, so nothing here is learned from our labels.

  python v6_embed.py octid octdl kermany
  python v6_embed.py octdl --blur 0.5 1           # embeddings of the blurred tiles (Step 5 TTA)
Writes data/cache/v6/emb_<net>_<ds>[_blur<s>].npy
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

from retina.v6data import CACHE, load_tiles

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def nets():
    import timm
    from torchvision import models
    r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    r.fc = torch.nn.Identity()
    d = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=(126, 504))
    return {"resnet18": (r.eval(), None), "dinov2s": (d.eval(), (126, 504))}


@torch.no_grad()
def embed(ds, blur, dev, bs=32):
    t = load_tiles(ds, blur)
    x_all = torch.from_numpy(np.where(t["valid"], t["hi"], 0).astype(np.float32) / 255.0)
    for name, (net, size) in nets().items():
        net = net.to(dev)
        out = []
        for i in range(0, len(x_all), bs):
            x = x_all[i:i + bs, None].repeat(1, 3, 1, 1)
            if size is not None:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            x = ((x - MEAN) / STD).to(dev)
            out.append(net(x).float().cpu().numpy())
        E = np.concatenate(out)
        f = CACHE / (f"emb_{name}_{ds}.npy" if not blur else f"emb_{name}_{ds}_blur{blur:g}.npy")
        np.save(f, E.astype(np.float32))
        print(ds, f"blur={blur}", name, E.shape, "->", f, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--blur", type=float, nargs="*", default=[0.0])
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = device(a.device)
    print("device", dev)
    for ds in a.datasets:
        for b in a.blur:
            embed(ds, b, dev)


if __name__ == "__main__":
    main()
