import numpy as np
S = np.load("/root/autodl-tmp/sid.npy", mmap_mode="r")
print("num_stations:", len(np.unique(S)))
