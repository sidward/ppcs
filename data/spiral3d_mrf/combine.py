import numpy as np
mps = np.zeros((10,) + (256,)*3, dtype=np.complex64)
for k in range(10):
  mps[k, ...] = np.load("mps_%03d.npy" % k, mmap_mode="r")
np.save("mps.npy", mps)
