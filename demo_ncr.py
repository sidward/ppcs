devnum   = 0             # Device to run optimization on. (-1) -> CPU.
ptol     = -float("inf") # l2-percentage tolerance between iterates.
norm     = "l_2"         # Cost function for polynomial optimization.
n_lamda  = 15            # Number of lambda values to test.
n_normal = 5             # Number of A.H * A evaluations per iteration to test.
max_iter = 80
verbose  = True

ksp_file = "data/spiral_brain/ksp.npy"
mps_file = "data/spiral_brain/mps.npy"
trj_file = "data/spiral_brain/trj.npy"

if __name__ == "__main__":

  import os
  if os.path.isdir("results/ncr"):
    import warnings
    warnings.warn("Directory 'results/ncr' exists. Continuing anyway...")
  os.makedirs("results/ncr/", exist_ok=True)

  import numpy as np
  import sigpy as sp
  import sigpy.mri as mr

  import utils
  import optalg
  import optpoly
  import prox

  device = sp.Device(devnum)
  xp = device.xp

  with device:

    np.random.seed(0)
    xp.random.seed(0)

    mvd = lambda x: sp.to_device(x, device)
    mvc = lambda x: sp.to_device(x, sp.cpu_device)

    mps = xp.load(mps_file)
    rss = xp.linalg.norm(mps, axis=0) > 0.5
    trj_ref = np.load(trj_file)

    b_ref = xp.load(ksp_file)
    b_ref /= xp.linalg.norm(b_ref)

    b_acq = b_ref[:, ::2, :]
    b_acq /= xp.linalg.norm(b_acq)

    trj_acq = trj_ref[::2, :, :]

    S = sp.linop.Multiply(mps.shape[1:], mps)
    F_ref = sp.linop.NUFFT(mps.shape, coord=trj_ref, toeplitz=True)
    F_acq = sp.linop.NUFFT(mps.shape, coord=trj_acq, toeplitz=True)
    W = sp.linop.Wavelet(S.ishape) * sp.linop.Multiply(S.ishape, rss)

    class A(sp.linop.Linop):
      def __init__(self, F, S):
        super().__init__(F.oshape, S.ishape)
        self.F = F
        self.S = S
      def _apply(self, input):
        return self.F(self.S(input))
      def _adjoint_linop(self):
        return self.S.H * self.F.H
      def _normal_linop(self):
        return self.S.H * self.F.N * self.S

    A_ref = A(F_ref, S)
    LL = sp.app.MaxEig(A_ref.N, dtype=xp.complex64, device=device,
                       show_pbar=False).run()
    A_ref = np.sqrt(1/LL) * A_ref

    A_acq = A(F_acq, S)
    LL = sp.app.MaxEig(A_acq.N, dtype=xp.complex64, device=device,
                       show_pbar=False).run()
    A_acq = np.sqrt(1/LL) * A_acq

    # Reference.
    loc = "results/ncr/reference"
    os.makedirs(loc, exist_ok=True)
    proxg = prox.L1Wav(A_acq.ishape, 6.5e-6, rss)
    ref = optalg.pgd(max_iter, ptol, A_ref, b_ref, proxg, precond_type=None,
                     save=loc, verbose=verbose)

    # CG.
    loc = "results/ncr/cg"
    os.makedirs(loc, exist_ok=True)
    optalg.cg(max_iter, ptol, A_acq, b_acq, ref=ref, save=loc, verbose=verbose)

    # FISTA.
    lst_lamda = [utils.lamda_helper((1e-2)/1.5**k)
                 for k in range(n_lamda)]
    for (lamda, sig, exp) in lst_lamda:
      proxg = prox.L1Wav(A_acq.ishape, lamda, rss)
      loc = "results/ncr/fista_%3.2fx10^%d" % (sig, exp)
      os.makedirs(loc, exist_ok=True)
      optalg.pgd(max_iter, ptol, A_acq, b_acq, proxg, precond_type=None,
                 save=loc, ref=ref, verbose=verbose)
    np.save("results/ncr/fista_params.npy", {"lst_lamda": lst_lamda})
    fista_params = utils.quality_filter("ncr", "fista", None, verbose=verbose)
    fista_lamda = fista_params[0] * 10**(fista_params[1])

    # IFISTA.
    lst_lamda = [utils.lamda_helper((1e-2)/1.5**k)
                 for k in range(n_lamda)]
    lst_normal = utils.normal_helper(max_iter, n_normal)
    for (lamda, sig, exp) in lst_lamda:
      proxg = prox.L1Wav(A_acq.ishape, lamda, rss)
      for normal in lst_normal:
        loc = "results/ncr/ifista_%3.2fx10^%d_%d" % (sig, exp, normal)
        os.makedirs(loc, exist_ok=True)
        optalg.pgd(int(max_iter/normal), ptol, A_acq, b_acq, proxg,
                   precond_type="ifista", pdeg=(normal - 1),
                   save=loc, ref=ref, verbose=verbose)
    np.save("results/ncr/ifista_params.npy", {"lst_lamda": lst_lamda,
                                              "lst_normal": lst_normal})

    # ADMM.
    lst_lamda = [utils.lamda_helper(1/3**(k - int(n_lamda/2)))
                 for k in range(n_lamda)]
    lst_normal = utils.normal_helper(max_iter, n_normal)
    proxg = prox.L1Wav(A_acq.ishape, fista_lamda, rss)
    for (lamda, sig, exp) in lst_lamda:
      for normal in lst_normal:
        loc = "results/ncr/admm_%3.2fx10^%d_%d" % (sig, exp, normal)
        os.makedirs(loc, exist_ok=True)
        optalg.admm(int(max_iter/normal), ptol, normal, A_acq, b_acq,
                    [proxg], lamda, ref=ref, save=loc,
                    verbose=verbose)
    np.save("results/ncr/admm_params.npy", {"lst_lamda": lst_lamda,
                                            "lst_normal": lst_normal})

    # PPCS.
    lst_lamda = [utils.lamda_helper((1e-2)/1.5**k)
                 for k in range(n_lamda)]
    lst_normal = utils.normal_helper(max_iter, n_normal)
    for (lamda, sig, exp) in lst_lamda:
      proxg = prox.L1Wav(A_acq.ishape, lamda, rss)
      for normal in lst_normal:
        loc = "results/ncr/pfista_%3.2fx10^%d_%d" % (sig, exp, normal)
        os.makedirs(loc, exist_ok=True)
        optalg.pgd(int(max_iter/normal), ptol, A_acq, b_acq, proxg,
                   precond_type="l_2", pdeg=(normal - 1), save=loc, ref=ref,
                   verbose=verbose)
    np.save("results/ncr/pfista_params.npy", {"lst_lamda": lst_lamda,
                                              "lst_normal": lst_normal})
