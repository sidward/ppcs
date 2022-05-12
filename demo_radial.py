devnum   = 0     # Device to run optimization on.
ptol     = 0.2   # Percentage tolerance between iterates.
pdeg     = 5     # Degree of polynomial to use.
norm     = "l_2" # Cost function for polynomial optimization.
l        = 0     # Smallest eigenvalue of A.H * A, if known.
lp_lamda = 1e-6  # Regularization value for LP.
pc_lamda = 2e-5  # Regularization value for PC.

ksp_file = 'data/radial/ksp.npy'
trj_file = 'data/radial/trj.npy'
mps_file = 'data/radial/mps.npy'

if __name__ == "__main__":

  import os
  if os.path.isdir("results/radial"):
    raise Exception("Directory 'results/radial' exists.")
  os.mkdir("results/radial/")

  import numpy as np
  import sigpy as sp
  import sigpy.mri as mr

  import optalg
  import optpoly

  trj = np.load(trj_file)

  device = sp.Device(devnum)
  xp = device.xp

  with device:

    mvd = lambda x: sp.to_device(x, device)
    mvc = lambda x: sp.to_device(x, sp.cpu_device)

    ksp = xp.load(ksp_file)
    mps = xp.load(mps_file)
    (nc, sy, sx) = mps.shape

    S = sp.linop.Multiply(mps.shape[1:], mps)
    F = sp.linop.NUFFT(mps.shape, coord=trj)
    W = sp.linop.Wavelet(S.ishape)
    A = F * S * W.H

    LL = sp.app.MaxEig(A.N, dtype=xp.complex64, \
                       device=device).run() * 1.01
    A = np.sqrt(1/LL) * A
    print("Maximum eigenvalue estimated:", LL)

    b = mvd(ksp)
    b = b/xp.linalg.norm(b)

    lp_proxg = sp.prox.L1Reg(A.ishape, lp_lamda)
    pc_proxg = sp.prox.L1Reg(A.ishape, pc_lamda)

    loc = "results/radial/lp"
    os.mkdir(loc)
    # Empty run.
    optalg.unconstrained(2, 1, A, b, lp_proxg, verbose=False)
    # Actual run.
    optalg.unconstrained(100, ptol, A, b, lp_proxg, save=loc)

    loc = "results/radial/pc"
    os.mkdir(loc)
    # Empty run.
    optalg.unconstrained(2, 1, A, b, pc_proxg, pdeg=pdeg, verbose=False)
    # Actual run.
    optalg.unconstrained(100, ptol, A, b, pc_proxg, save=loc, pdeg=pdeg)
