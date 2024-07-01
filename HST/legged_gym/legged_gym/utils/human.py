import numpy as np
import torch

def load_target_jt(device, file, offset):
    one_target_jt = np.load(f"data/{file}").astype(np.float32)
    one_target_jt = torch.from_numpy(one_target_jt).to(device)
    target_jt = one_target_jt.unsqueeze(0)
    target_jt += offset

    size = torch.tensor([one_target_jt.shape[0]]).to(device)
    return target_jt, size

