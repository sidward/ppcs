devnum = 0
rho = 1e-4
eps = 0.1525
num_iters = 201
cg_num_normal = [2, 3]
pi_num_normal = [1, 2]

ksp_file = 'data/cardiac/ksp.npy'
trj_file = 'data/cardiac/trj.npy'
mps_file = 'data/cardiac/mps.npy'

if __name__ == "__main__":

  import os
  if os.path.isdir("results/cardiac_constrained"):
    raise Exception("Directory 'results/cardiac_constrained' exists.")
  os.mkdir("results/cardiac_constrained/")

  import numpy as np
  import sigpy as sp
  import sigpy.mri as mr

  import optalg
  import optpoly

  ksp = np.load(ksp_file)
  trj = np.load(trj_file)

  device = sp.Device(devnum)
  xp = device.xp

  with device:

    mvd = lambda x: sp.to_device(x, device)
    mvc = lambda x: sp.to_device(x, sp.cpu_device)

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

    g = lambda x: xp.linalg.norm(x.ravel(), ord=1)
    proxg = sp.prox.L1Reg(A.ishape, 1)

    # Empty run.
    optalg.constrained(2, 2, A, b, eps, proxg, rho, method="cg")
    # Actual run.
    for cg_nn in cg_num_normal:
      loc = "results/cardiac_constrained/cg_nn_%02d" % (cg_nn)
      os.mkdir(loc)
      optalg.constrained(num_iters, cg_nn, A, b, eps, proxg, rho,
                         g=g, method="cg", save=loc, verbose=True)

    # Empty run.
    optalg.constrained(2, 2, A, b, eps, proxg, rho, method="pi")
    # Actual run.
    for pi_nn in pi_num_normal:
      loc = "results/cardiac_constrained/pi_nn_%02d" % (pi_nn)
      os.mkdir(loc)
      optalg.constrained(num_iters, pi_nn, A, b, eps, proxg, rho,
                         g=g, method="pi", save=loc, verbose=True)

