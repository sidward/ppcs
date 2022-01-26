devnum = 0
num_iters = 201
rho = 1
lamda_1 = 1e-5
lamda_2 = 5e-6
cg_num_normal = [2, 3]
pi_num_normal = [1, 2]

ksp_file = "data/cardiac/ksp.npy"
trj_file = "data/cardiac/trj.npy"
mps_file = "data/cardiac/mps.npy"

if __name__ == "__main__":

  import os
  if os.path.isdir("results/cardiac_consensus"):
    raise Exception("Directory 'results/cardiac_consensus' exists.")
  os.mkdir("results/cardiac_consensus/")

  import numpy as np
  import sigpy as sp
  import sigpy.mri as mr

  import optalg
  import optpoly
  import tv

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
    F = sp.linop.NUFFT(mps.shape, coord=trj, toeplitz=True)
    A = F * S

    LL = sp.app.MaxEig(A.N, dtype=xp.complex64, \
                       device=device).run() * 1.01
    A = np.sqrt(1/LL) * A
    print("Maximum eigenvalue estimated:", LL)

    b = mvd(ksp)
    b = b/xp.linalg.norm(b)

    W1 = sp.linop.Wavelet(S.ishape, wave_name="db4")
    g1 = lambda x: lamda_1 * xp.linalg.norm(W1(x).ravel(), ord=1)
    prox_g1 = sp.prox.UnitaryTransform( \
              sp.prox.L1Reg(W1.oshape, lamda_1), W1)

    prox_g2 = tv.ProxTV(A.ishape, lamda_2)
    g2 = lambda x: lamda_2 * xp.linalg.norm(prox_g2.G(x))

    lst_g = [g1, g2]
    lst_proxg = [prox_g1, prox_g2]

    # Empty run.
    optalg.consensus(2, 2, A, b, lst_proxg, rho, lst_g=lst_g, method="cg")
    # Actual run.
    for cg_nn in cg_num_normal:
      loc = "results/cardiac_consensus/cg_nn_%02d" % (cg_nn)
      os.mkdir(loc)
      optalg.consensus(num_iters, cg_nn, A, b, lst_proxg, rho,
                       lst_g=lst_g, method="cg", save=loc, verbose=True)

    # Empty run.
    optalg.consensus(2, 2, A, b, lst_proxg, rho, lst_g=lst_g, method="pi")
    # Actual run.
    for pi_nn in pi_num_normal:
      loc = "results/cardiac_consensus/pi_nn_%02d" % (pi_nn)
      os.mkdir(loc)
      optalg.consensus(num_iters, pi_nn, A, b, lst_proxg, rho,
                       lst_g=lst_g, method="pi", save=loc, verbose=True)
