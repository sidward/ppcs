devlst   = [0, 1, 2]     # Device to run optimization on. (-1) -> CPU.
ptol     = -float("inf") # l2-percentage tolerance between iterates.
norm     = "l_2"         # Cost function for polynomial optimization.
alpha    = 1e3           # ADMM step size.
fista_l  = 4.5e-5        # FISTA regularization.
pfista_l = 7.5e-4        # Poly. Precond. FISTA regularization.
pfista_d = 4             # Polynomial degree.
in_iter  = 40            # Maximum number of AHA for the inner iterations.
out_iter = 2             # Number of ADMM outer iterations.
use_poly = True
verbose  = True

ksp_file = "data/spiral3d_mrf/ksp.npy"
trj_file = "data/spiral3d_mrf/trj.npy"
mps_file = "data/spiral3d_mrf/mps.npy"
phi_file = "data/spiral3d_mrf/phi.npy"
dcf_file = "data/spiral3d_mrf/dcf.npy"

if __name__ == "__main__":

  import os
  import time
  os.makedirs("results/mrf", exist_ok=True)

  import numpy as np
  import sigpy as sp
  import sigpy.mri as mr
  import multiprocessing as mp
  from sigpy.mri.dcf import pipe_menon_dcf

  import utils
  import optalg
  import optpoly
  import prox

  np.random.seed(0)

  phi = np.load(phi_file)
  phi = phi @ sp.fft(np.eye(phi.shape[-1]), axes=(0,))
  phi = phi.T

  mps = np.load(mps_file)
  rss = np.linalg.norm(mps, axis=0) > 0.5

  b = np.transpose(np.load(ksp_file, mmap_mode="r"), (1, 2, 0, 3)).T
  b = np.transpose(b, (1, 0, 2, 3))
  b = b/np.linalg.norm(b)

  trj = 256 * np.load(trj_file)[:, 10:, :, :]
  trj = trj[::-1, ...].T

  if not os.path.isfile(dcf_file):
    dcf = sp.to_device(pipe_menon_dcf(trj, img_shape=mps.shape[1:],
                       device=devlst[0], show_pbar=True).real,
                       sp.cpu_device)
    dcf /= np.linalg.norm(dcf.ravel(), ord=np.inf)
    np.save(dcf_file, dcf)
    print("--> DCF saved. Please run script again.")
    exit(0)
  else:
    dcf = np.load(dcf_file)
  dcf = np.sqrt(dcf)

  b *= dcf[None, ...]
  b = b/np.linalg.norm(b)

  N   = len(devlst)
  nc  = N * int(b.shape[0]/N)
  b   = b[:nc, ...]
  mps = mps[:nc, ...]
  
  def subrecon(args):
    (idx, use_poly, x0, v) = args
    devnum = devlst[idx]

    _b   = np.split(b,   N, axis=0)[idx]
    _mps = np.split(mps, N, axis=0)[idx]

    device = sp.Device(devnum)
    xp = device.xp

    with device:
      p_d = sp.to_device(dcf, device)
      p_p = sp.to_device(phi, device)
      p_r = sp.to_device(rss, device)

      p_m = sp.to_device(_mps, device)

      F = sp.linop.Multiply(_b.shape[1:], p_d) * \
          sp.linop.NUFFT(p_m.shape[1:], trj)

      outer_A = []
      for k in range(p_m.shape[0]):
        S = sp.linop.Multiply(p_m.shape[1:], p_m[k, ...]) * \
            sp.linop.Reshape(p_m.shape[1:], [1] + list(p_m.shape[1:]))

        lst_A = [sp.linop.Reshape([1] + list(F.oshape), F.oshape) * \
                 sp.linop.Multiply(F.oshape, p_p[k, :, None, None]) * \
                 F * S for k in range(p_p.shape[0])]

        inner_A = sp.linop.Hstack(lst_A, axis=0)
        outer_A.append(inner_A)

      # 10 coil maximum eigenvalue.
      LL = 0.0095
      A = (1/LL) * sp.linop.Vstack(outer_A, axis=0)

      # Subset A eigenvalue.
      LL = (60.84, 61.10, 60.18,)[idx]

      # Sub problem.
      I = (1/(alpha**0.5)) * sp.linop.Identity(A.ishape)
      P = LL + (1/alpha)
      T = (1/(P**0.5)) * sp.linop.Vstack((A, I))
      lamda = pfista_l/(N * (P**(0.5))) if use_poly else \
               fista_l/(N * (P**(0.5)))
      proxg = prox.LLR(A.ishape, lamda, 8, p_r)

      t = sp.to_device(np.concatenate((_b.ravel(), v.ravel()/(alpha**0.5)),
                                      axis=0), device)/(P**0.5)

      save = None
      if idx == 0:
        save = f"{loc}/subproblem"
        os.makedirs(save, exist_ok=True)

      if use_poly:
        x = optalg.pgd(in_iter//(pfista_d + 1), -np.inf, T, t, proxg, x0=x0,
                       precond_type="l_2", pdeg=pfista_d,
                       verbose=(idx == 0), save=save)
      else:
        x = optalg.pgd(in_iter, -np.inf, T, t, proxg, x0=x0,
                       verbose=(idx == 0), save=save)
      x = sp.to_device(x, sp.cpu_device)

    return x

  if use_poly:
    (pfista_l, sig, exp) = utils.lamda_helper(pfista_l)
    loc = "results/mrf/pfista_%3.2fx10^%d_%d" % (sig, exp, pfista_d + 1)
  else:
    (fista_l, sig, exp) = utils.lamda_helper(fista_l)
    loc = "results/mrf/fista_%3.2fx10^%d" % (sig, exp)
  os.makedirs(loc, exist_ok=True)

  with mp.Pool(N) as p:
    x  = np.zeros((   5,) + (256,)*3, dtype=np.complex64)
    ui = np.zeros((N, 5,) + (256,)*3, dtype=np.complex64)

    lst_t = []
    for k in range(out_iter):
      start_time = time.perf_counter()
      vi   = x[None, ...] - ui
      args = [(d, use_poly, x.copy(), vi[d, ...]) for d in range(N)]
      xi   = np.stack(p.map(subrecon, args))
      x    = np.mean(xi, axis=0)
      ui   = ui + xi - x[None, ...]
      end_time = time.perf_counter()

      lst_t.append(end_time - start_time)
      print("==> ADMM Iteration %d done. Time taken: %f." % (k + 1, lst_t[-1]))

      np.save(f"{loc}/time.npy", np.array(lst_t))
      np.save(f"{loc}/iter_%03d.npy" % (k), x)
