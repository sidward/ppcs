import numpy as np
import sigpy as sp
import time
import optpoly

from tqdm.auto import tqdm

def cg(num_iters, ptol, A, b, P=None, lamda=None, ref=None,
       save=None, verbose=True, idx=None):
  r"""Conjugate Gradient.

  Solves for the following optimization problem.

  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2

  Inputs:
    num_iters (Int): Maximum number of iterations.
    ptol (Float): l1-percentage tolerance between iterates.
    A (Linop): Forward model.
    b (Array): Measurement.
    P (None or Linop): Preconditioner.
    ref (None or Array): Reference to compare against.
    lamda (None or Float): l2-regularization.
    save (None or String): If specified, path to save iterations and
                           timings.
    verbose (Bool): Print information.
    idx (Tuple): If passed, slice iterates before saving.

  Returns:
    x (Array): Reconstruction.
  """
  device = sp.get_device(b)
  xp = device.xp

  with device:

    lst_time = []
    lst_err  = None
    if type(ref) != type(None):
      ref = sp.to_device(ref, device)
      lst_err = ([], [])

    AHA = A.N
    if lamda is not None:
      AHA = AHA + lamda * sp.linop.Identitiy(A.ishape)

    if P == None:
      P = sp.linop.Identity(A.ishape)

    # Set-up time.
    start_time = time.perf_counter()
    AHb = A.H(b)
    x = xp.zeros_like(AHb)
    r = AHb - AHA(x)
    z = P(r.copy())
    p = z.copy()
    rzold = xp.real(xp.vdot(r, z))
    end_time = time.perf_counter()

    lst_time.append(end_time - start_time)
    if lst_err != None:
      lst_err[0].append(calc_perc_err(ref, x, ord=1))
      lst_err[1].append(calc_perc_err(ref, x, ord=2))

    save_helper(save, x, 0, lst_time, lst_err, idx)

    if verbose:
      pbar = tqdm(total=num_iters, desc="CG",
                  leave=True)
    for k in range(0, num_iters):
      x_old = x.copy()
      start_time = time.perf_counter()

      Ap = AHA(p)
      pAp = xp.real(xp.vdot(p, Ap)).item()
      if pAp > 0:
        alpha = rzold / pAp
        sp.axpy(x, alpha, p)
        sp.axpy(r, -alpha, Ap)
        z = P(r.copy())
        rznew = xp.real(xp.vdot(r, z))
        beta = rznew / rzold
        sp.xpay(p, beta, z)
        rzold = rznew

      end_time = time.perf_counter()

      lst_time.append(end_time - start_time)
      if lst_err != None:
        lst_err[0].append(calc_perc_err(ref, x, ord=1))
        lst_err[1].append(calc_perc_err(ref, x, ord=2))
      save_helper(save, x, k + 1, lst_time, lst_err, idx)

      calc_tol = calc_perc_err(x_old, x, ord=1)
      if verbose:
        pbar.set_postfix(ptol="%0.2f%%" % calc_tol)
        pbar.update()
        pbar.refresh()

      if pAp <= 0 or calc_tol <= ptol:
        break;

    if verbose:
      pbar.close()
    return x

def pgd(num_iters, ptol, A, b, proxg, x0=None, precond_type=None,
        pdeg=None, accelerate=True, l=0, stepfn=None,
        ref=None, save=None, verbose=True, idx=None):
  r"""Proximal Gradient Descent.

  Solves for the following optimization problem using proximal gradient
  descent:

  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2 + g(x)

  Assumes MaxEig(A.H * A) = 1.

  Inputs:
    num_iters (Int): Maximum number of iterations.
    ptol (Float): l1-percentage tolerance between iterates.
    A (Linop): Forward model.
    b (Array): Measurement.
    proxg (Prox): Proximal operator of g.
    x0 (Array): Initial guess.
    precond_type (String): Type of preconditioner.
                           - "l_2"    = l_2 optimized polynomial.
                           - "l_inf"  = l_inf optimized polynomial.
                           - "ifista" = from DOI: 10.1137/140970537.
    pdeg (None or Int): Degree of polynomial preconditioner to use.
                        If None, do not use preconditioner.
    accelerate (Bool): If True, use Nestrov acceleration.
    l (Float): If known, minimum eigenvalue of A.H * A.
    stepfn (function): If specified, this function determines the variable
                       step size per iteration.
    ref (None or Array): Reference to compare against.
    save (None or String): If specified, path to save iterations and
                           timings.
    verbose (Bool): Print information.
    idx (Tuple): If passed, slice iterates before saving.

  Returns:
    x (Array): Reconstruction.
  """
  if precond_type == "l_inf" and l == 0:
    raise Exception("If l == 0, l_inf polynomial cannot be used.")

  device = sp.get_device(b)
  xp = device.xp

  if verbose:
    print("Proximal Gradient Descent.")
    if accelerate:
      print("> Variable step size." if l == 0 else \
            "> Fixed step size derived from l = %5.2f" % l)
    if precond_type == None:
      print("> No preconditioning used.")
    else:
      print("> %s-preconditioning is used." % precond_type)

  P = sp.linop.Identity(A.ishape) if pdeg is None else  \
      optpoly.create_polynomial_preconditioner(precond_type, pdeg, A.N, l, 1,
                                               verbose=verbose)

  with device:

    lst_time = []
    lst_err  = None
    if type(ref) != type(None):
      ref = sp.to_device(ref, device)
      lst_err = ([], [])

    # Set-up time.
    start_time = time.perf_counter()
    AHb = A.H(b)
    x = AHb if x0 is None else sp.to_device(x0, device)
    z = x.copy()
    end_time = time.perf_counter()

    lst_time.append(end_time - start_time)
    if lst_err != None:
      lst_err[0].append(calc_perc_err(ref, x, ord=1))
      lst_err[1].append(calc_perc_err(ref, x, ord=2))
    save_helper(save, x, 0, lst_time, lst_err, idx)

    if verbose:
      pbar = tqdm(total=num_iters, desc="PGD", leave=True)
    for k in range(0, num_iters):
      start_time = time.perf_counter()
        
      x_old = x.copy()
      if accelerate:
        x = z.copy()
        
      gr = A.N(x) - AHb
      x = proxg(1, x - P(gr))
            
      if accelerate:
        if l > 0:
          # DOI: 10.1007/978-3-319-91578-4_2
          step = (1 - l**(0.5))/(1 + l**(0.5))
        elif stepfn == None:
          # DOI: 10.1561/2400000003
          step = k/(k + 3)
        else:
          step = stepfn(k)
        z = x + step * (x - x_old)

      end_time = time.perf_counter()

      lst_time.append(end_time - start_time)
      if lst_err != None:
        lst_err[0].append(calc_perc_err(ref, x, ord=1))
        lst_err[1].append(calc_perc_err(ref, x, ord=2))
      save_helper(save, x, k + 1, lst_time, lst_err, idx)

      calc_tol = calc_perc_err(x_old, x, ord=1)
      if verbose:
        pbar.set_postfix(ptol="%0.2f%%" % calc_tol)
        pbar.update()
        pbar.refresh()

      if calc_tol <= ptol:
        break;

    if verbose:
      pbar.close()
    return x

def admm(num_iters, ptol, num_normal, A, b, lst_proxg, rho,
         lst_g=None, method="cg", P=None, ref=None,
         save=None, verbose=True, idx=None):
  r"""ADMM.

  Solves for the following optimization problem:

  .. math::
    \min_x \frac{1}{2} \| A x - y \|_2^2 + \sum_i g_i(x)

  Each iteration involves solving an inner least squares problem which
  is parameterized by the number of A.H * A evaluations.

  Based on:
    Parikh, N., & Boyd, S.
    Proximal algorithms.
    Foundations and Trends in optimization, 1(3), 127-239.
    DOI: 10.1561/2400000003

  Assumes MaxEig(A.H * A) = 1.

  Inputs:
    num_iters (Int): Number of ADMM iterations.
    ptol (Float): l1-percentage tolerance between iterates.
    num_normal (Int): Number of A.H * A evaluations.
    A (Linop): Forward linear operator.
    b (Array): Measurement.
    lst_proxg (List of Prox): List of proximal operators.
    rho (Float): ADMM step size.
    lst_g (List of Functions): List of regularizations. If specified,
                               objective over iterations are saved.
    method (String): Determines the method used to solve for the inner
                     least squares.
                     - "cg": Conjugate gradient.
                     - "pi": Polynomial inversion.
    P (None or Linop): Preconditioner for "cg".
    save (None or String): If specified, path to save iterations, costs and
                           timings.
    verbose (Bool): Print information.
    idx (Tuple): If passed, slice iterates before saving.

  Returns:
    x (Array): Reconstruction.
  """
  device = sp.get_device(b)
  xp = device.xp

  assert num_normal >= 1
  assert method == "cg" or method == "pi"

  if num_normal == 1 and method == "cg":
    raise Exception("CG requires >= 2 normal evaluations.")

  c = len(lst_proxg)
  def calculate_objective(x):
    obj = sp.to_device(0.5 * xp.linalg.norm(A * x - b)**2, sp.cpu_device)
    for j in range(c):
      obj += sp.to_device(lst_g[j](x), sp.cpu_device)
    return obj

  with device:

    lst_time = []
    lst_err = None
    if type(ref) != type(None):
      ref = sp.to_device(ref, device)
      lst_err = ([], [])

    stats = lst_g is not None and save is not None
    if stats:
      lst_obj = []

    start_time = time.perf_counter()
    AHb = A.H(b)
    x = AHb.copy()
    r_AHb = rho * AHb
    xi = xp.zeros([1 + c] + list(x.shape), dtype=xp.complex64)
    ui = xp.zeros_like(xi)
    end_time = time.perf_counter()

    lst_time.append(end_time - start_time)
    if lst_err != None:
      lst_err[0].append(calc_perc_err(ref, x, ord=1))
      lst_err[1].append(calc_perc_err(ref, x, ord=2))
    if stats:
      lst_obj.append(calculate_objective(x))
      save_helper(save, x, 0, lst_time, lst_err, idx, lst_obj)
    else:
      save_helper(save, x, 0, lst_time, lst_err, idx)

    T = sp.linop.Identity(x.shape) + rho * A.N
    if method == "pi":
      P = optpoly.create_polynomial_preconditioner(num_normal - 1, T, 1,
                                                   1 + rho, norm="l_inf",
                                                   verbose=verbose)

    def prox_f(rho, v):
      if method == "cg":
        sp.app.App(sp.alg.ConjugateGradient(T, v + r_AHb, v, P=P,
                                            max_iter=num_normal - 1,
                                            tol=1e-4), leave_pbar=False,
                                            show_pbar=False,
                                            record_time=False).run()
      else:
        v = v - P * (T(v) - (v + r_AHb))
      return v
    
    if verbose:
      pbar = tqdm(total=num_iters, desc="ADMM", leave=True)
    for k in range(num_iters):
      x_old = x.copy()
      start_time = time.perf_counter()

      xi[0, ...] = prox_f(rho, x - ui[0, ...])
      for j in range(c):
        xi[1 + j, ...] = lst_proxg[j](rho, x - ui[1 + j, ...])

      x = xp.mean(xi, axis=0)
      ui += xi - x[None, ...]

      end_time = time.perf_counter()

      lst_time.append(end_time - start_time)
      if lst_err != None:
        lst_err[0].append(calc_perc_err(ref, x, ord=1))
        lst_err[1].append(calc_perc_err(ref, x, ord=2))
      if stats:
        lst_obj.append(calculate_objective(x))
        save_helper(save, x, k + 1, lst_time, lst_err, idx, lst_obj)
      else:
        save_helper(save, x, k + 1, lst_time, lst_err, idx)

      calc_tol = calc_perc_err(x_old, x, ord=1)
      if verbose:
        pbar.set_postfix(ptol="%0.2f%%" % calc_tol)
        pbar.update()
        pbar.refresh()

      if calc_tol <= ptol:
        break;

    if verbose:
      pbar.close()
    return x

def calc_perc_err(ref, x, ord=2, auto_normalize=True):
  dev = sp.get_device(x)
  xp = dev.xp
  with dev:
    if auto_normalize:
      p = ref/xp.linalg.norm(ref.ravel(), ord=ord)
      q =   x/xp.linalg.norm(  x.ravel(), ord=ord)
      err = xp.linalg.norm((p - q).ravel(), ord=ord)
    else:
      err = xp.linalg.norm((ref - x).ravel(), ord=ord)
      err /= xp.linalg.norm(ref.ravel(), ord=ord)
    err = sp.to_device(err, sp.cpu_device)
  if np.isnan(err) or np.isinf(err):
    return 100
  return 100 * err

def save_helper(save, x, itr, lst_time, lst_err, idx, obj=None):
  if save == None:
    return

  tp = sp.get_array_module(x)
  np.save("%s/time.npy" % save, np.cumsum(lst_time))
  if idx is None:
    tp.save("%s/iter_%03d.npy" % (save, itr), x)
  else:
    tp.save("%s/iter_%03d.npy" % (save, itr), x[idx])
  if obj is not None:
    np.save("%s/obj.npy" % save, lst_obj)
  if lst_err is not None:
    np.save("%s/err.npy" % save, lst_err)
