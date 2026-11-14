import time, torch, numpy as np, sys
sys.path.insert(0, ".")
from v6_cnn import Scans, net_for
from torch.utils.data import DataLoader
from retina.v6run import task_data
d = task_data("OCTID CSR/DR/MH (500x750)", ())
classes = list(np.unique(d["y"]))
ds = Scans(d["paths"][:128], d["y"][:128], classes, True)
t=time.time(); xs=[ds[i] for i in range(64)]; print("decode 64 imgs (main proc)", round(time.time()-t,2))
for dev in ["mps", "cpu"]:
    net = net_for(3).to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-4)
    xb = torch.stack([x for x,_ in xs[:32]]).to(dev); yb = torch.tensor([y for _,y in xs[:32]]).to(dev)
    for i in range(6):
        if i == 1: t=time.time()
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(net(xb), yb); l.backward(); opt.step()
    if dev=="mps": torch.mps.synchronize()
    print(dev, "s/step bs32", round((time.time()-t)/5, 3))
