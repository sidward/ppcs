devnum   = 0       # Device to run optimization on.
ptol     = 0.2     # Percentage tolerance between iterates.
pdeg     = 9       # Degree of polynomial to use.
norm     = "l_2"   # Cost function for polynomial optimization.
l        = 0       # Smallest eigenvalue of A.H * A, if known.
blk      = 8       # Block size for locally low-rank.
lp_lamda = 2.5e-6  # Regularization value for LP.
pc_lamda = 8.75e-5 # Regularization value for PC.

ksp_file = "data/mrf3d/ksp.npy"
trj_file = "data/mrf3d/trj.npy"
phi_file = "data/mrf3d/phi.npy"
mps_file = "data/mrf3d/mps.npy"

if __name__ == "__main__":

  import os

  if not os.path.isdir("results/mrf3d"):
    os.mkdir("results/mrf3d")

  if os.path.isdir("results/mrf3d/lp_%s" % repr(lp_lamda)):
    raise Exception("Directory 'results/mrf3d/lp_%s' exists." % \
                     repr(lp_lamda))

  if os.path.isdir("results/mrf3d/pc_%s" % repr(pc_lamda)):
    raise Exception("Directory 'results/mrf3d/pc_%s' exists." % \
                     repr(pc_lamda))

  os.mkdir("results/mrf3d/lp_%s" % repr(lp_lamda))
  os.mkdir("results/mrf3d/pc_%s" % repr(pc_lamda))

  import numpy as np
  import sigpy as sp
  import sigpy.mri as mr

  import optalg
  import optpoly
  import llr

  ksp = np.transpose(np.load(ksp_file, mmap_mode="r"), (1, 2, 0, 3))
  ksp = ksp[10:, :, :, :].T
  ksp = np.transpose(ksp, (1, 0, 2, 3))

  trj = np.load(trj_file)[:, 10:, :, :]
  trj[0, ...] = trj[0, ...] * 256
  trj[1, ...] = trj[1, ...] * 256
  trj[2, ...] = trj[2, ...] * 256
  trj = trj[::-1, ...].T

  phi = np.load(phi_file)
  mps = np.load(mps_file)

  device = sp.Device(devnum)
  xp = device.xp

  with device:

    mvd = lambda x: sp.to_device(x, device)
    mvc = lambda x: sp.to_device(x, sp.cpu_device)

    phi = mvd(phi)
    mps = mvd(mps)

    F = sp.linop.NUFFT(mps.shape[1:], trj)
    outer_A = []
    for k in range(mps.shape[0]):
      S = sp.linop.Multiply(mps.shape[1:], mps[k, ...]) * \
          sp.linop.Reshape( mps.shape[1:], [1] + list(mps.shape[1:]))
      lst_A = [sp.linop.Reshape([1] + list(F.oshape), F.oshape)   * \
               sp.linop.Multiply(F.oshape, phi[k, :, None, None]) * \
               F * S for k in range(phi.shape[0])]
      inner_A = sp.linop.Hstack(lst_A, axis=0)
      D1 = sp.linop.ToDevice(inner_A.ishape, device, sp.cpu_device)
      D2 = sp.linop.ToDevice(inner_A.oshape, sp.cpu_device, device)
      outer_A.append(D2 * inner_A * D1) 
    A = sp.linop.Vstack(outer_A, axis=0)

    LL = sp.app.MaxEig(A.N, dtype=xp.complex64, \
                       device=sp.cpu_device).run() * 1.01
    print("Maximum eigenvalue estimated:", LL)
    A = np.sqrt(1/LL) * A

    b = ksp/np.linalg.norm(ksp)

    lp_proxg = llr.ProxLLR(A.ishape, lp_lamda, blk)
    pc_proxg = llr.ProxLLR(A.ishape, pc_lamda, blk)
    
    idx = (slice(None), 160, slice(None), slice(None))

    # Empty run.
    optalg.unconstrained(2, 1, A, b, lp_proxg, idx=idx, verbose=False)

    # Actual run.
    loc = "results/mrf3d/lp_%s" % repr(lp_lamda)
    optalg.unconstrained(100, ptol, A, b, lp_proxg, save=loc, idx=idx)

    # Empty run.
    optalg.unconstrained(2, 1, A, b, pc_proxg, pdeg=pdeg, idx=idx, \
                         verbose=False)

    # Actual run.
    loc = "results/mrf3d/pc_%s" % repr(pc_lamda)
    optalg.unconstrained(100, ptol, A, b, pc_proxg, save=loc, \
                         pdeg=pdeg, idx=idx)
