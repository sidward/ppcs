import numpy as np
import sigpy as sp
import time
import optpoly

from tqdm.auto import tqdm

def unconstrained(num_iters, ptol, A, b, proxg, pdeg=None,
        norm="l_2", accelerate=True, l=0, stepfn=None,
        save=None, verbose=True, idx=None):
  r"""Unconstrained Optimization.

  Solves for the following optimization problem using proximal gradient
  descent:

  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2 + g(x)

  Assumes MaxEig(A.H * A) = 1.

  If polynomial preconditioning is used, note the the final solution is a
  close approximation to the true solution.

  Inputs:
    num_iters (Int): Maximum number of iterations.
    ptol (Float): Percentage tolerance between iterates.
    A (Linop): Forward model.
    b (Array): Measurement.
    proxg (Prox): Proximal operator of g.
    pdeg (None or Int): Degree of polynomial preconditioner to use.
                        If None, do not use preconditioner.
    norm (String): Norm to optimize. Currently only supports "l_2" and
                   "l_inf".
    accelerate (Bool): If True, use Nestrov acceleration.
    l (Float): Minimum eigenvalue of A.H * A.
    stepfn (function): If specified, this function determines the variable
                       step size per iteration.
    save (None or String): If specified, path to save iterations and
                           timings.
    verbose (Bool): Print information.
    idx (Tuple): If passed, slice iterates before saving.

  Returns:
    x (Array): Reconstruction.
  """
  if norm == "l_inf" and l == 0:
    raise Exception("If l == 0, l_2 norm must be used.")

  device = sp.get_device(b)
  xp = device.xp

  if verbose:
    print("Unconstrained Optimization.")
    if accelerate:
      print("> Variable step size." if l == 0 else \
            "> Fixed step size derived from l = %5.2f" % l)
    print("> Preconditioning is%sused." % \
         (" " if pdeg is not None else " not "))

  P = sp.linop.Identity(A.ishape) if pdeg is None else  \
      optpoly.create_polynomial_preconditioner(pdeg, A.N, l, 1, \
                                               norm=norm, verbose=verbose)

  with device:

    AHb = A.H(b)
    x = AHb.copy()
    z = x.copy()

    lst_time  = []
    calc_tol = -1
    if verbose:
      pbar = tqdm(total=num_iters, desc="Unconstrained Optimization", \
                  leave=True)
    for k in range(1, num_iters + 1):
      start_time = time.perf_counter()
        
      x_old = x.copy()
      if accelerate:
        x = z.copy()
        
      gr = A.N(x) - AHb
      x = proxg(1, x - P(gr))
            
      if accelerate:
        if l == 0:
          # DOI: 10.1007/978-3-319-91578-4_2
          step = (1 - l**(0.5))/(1 + l**(0.5))
        elif stepfn == None:
          # DOI: 10.1007/s10957-015-0746-4
          step = (k - 1)/(k + 4)
        else:
          step = stepfn(k)
        z = x + step * (x - x_old)

      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)

      calc_tol = 100 * xp.linalg.norm(x_old - x)/xp.linalg.norm(x)
      if calc_tol <= ptol:
        break;

      if save != None:
        tp = sp.get_array_module(x)
        np.save("%s/time.npy" % save, np.cumsum(lst_time))
        if idx is None:
          tp.save("%s/iter_%03d.npy" % (save, k), x)
        else:
          tp.save("%s/iter_%03d.npy" % (save, k), x[idx])

      if verbose:
        pbar.set_postfix(ptol="%0.2f%%" % calc_tol)
        pbar.update()
        pbar.refresh()

    if verbose:
      pbar.set_postfix(ptol="%0.2f%%" % calc_tol)
      pbar.close()
    return x

def constrained(num_iters, num_normal, A, b, eps, proxg, lamda,
                g=None, method="cg", save=None, verbose=False):
  r"""Constrained Optimization.

  Solves for the following optimization problem:

  .. math::
    \min_x g(x)
    s.t. \| A x - b \|_2 < \epsilon

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
    num_normal (Int): Number of A.H * A evaluations.
    A (Linop): Forward linear operator.
    b (Array): Measurement.
    eps (Float): Constraint value.
    proxg (Prox): Proximal operator of g.
    lamda (Float): ADMM step size.
    g (Function): Objective. If specified, objective over iteration
                  and data consistency error over iteration are saved. 
    method (String): Determines the method used to solve for the inner
                     least squares.
                     - "cg": Conjugate gradient.
                     - "pi": Polynomial inversion.
    save (None or String): If specified, path to save iterations, costs and
                           timings.
    verbose (Bool): Print information.

  Returns:
    x (Array): Reconstruction.
  """
  device = sp.get_device(b)
  xp = device.xp

  assert num_normal >= 1
  assert method == "cg" or method == "pi"

  if num_normal == 1 and method == "cg":
    raise Exception("CG requires >= 2 normal evaluations.")

  lst_time = []
  stats = g is not None and save is not None
  if stats:
    lst_obj  = []
    lst_dc   = []

  with device:

    n = int(np.prod(A.ishape))
    m = int(np.prod(A.oshape))

    x = xp.zeros((n + m,), dtype=xp.complex64)

    x1 = x.copy()
    x2 = x.copy()

    u1 = xp.zeros_like(x)
    u2 = xp.zeros_like(x)

    Ri = sp.linop.Reshape(A.ishape, (n,))
    Ro = sp.linop.Reshape(A.oshape, (m,))

    T = sp.linop.Vstack([sp.linop.Identity((n,)), Ro.H * A * Ri])
    if method == "pi":
      P = optpoly.create_polynomial_preconditioner(num_normal - 1, T.N, \
                                                   1, 2, norm="l_inf",  \
                                                   verbose=verbose)

    def prox_f1(lamda, v):
      v1 = v[:n]
      v2 = v[n:]

      v1 = Ri.H(proxg(lamda, Ri(v1)))

      vec = v2 - Ro.H(b)
      nrm = xp.linalg.norm(vec)
      if nrm > eps:
        v2 = Ro.H(b) + eps * (vec/nrm)

      v[:n] = v1
      v[n:] = v2

      return v

    def prox_f2(lamda, v):
      THv = T.H * v
      v1  = v[:n]
      if method == "cg":
        sp.app.App(sp.alg.ConjugateGradient(T.N, THv, v1,             \
                                            max_iter=num_normal - 1,  \
                                            tol=1e-4),                \
                                            leave_pbar=False,         \
                                            show_pbar=False,          \
                                            record_time=False).run()
      else:
        v1 = v1 - P * (T.N(v1) - THv)

      v[:n] = v1
      v[n:] = Ro.H(A(Ri(v1)))

      return v
    
    pbar = tqdm(total=num_iters, desc="Constrained Optimization", \
                leave=True)
    for k in range(num_iters):
      start_time = time.perf_counter()

      x1 = prox_f1(lamda, x - u1)
      x2 = prox_f2(lamda, x - u2)

      x = (x1 + x2)/2

      u1 += x1 - x
      u2 += x2 - x

      end_time = time.perf_counter()

      dt = end_time - start_time
      lst_time.append(dt)

      if stats:
        err = sp.to_device(xp.linalg.norm(A * Ri(x[:n]) - b), \
                           sp.cpu_device)
        obj = sp.to_device(g(Ri(x[:n])), sp.cpu_device)
        lst_dc.append(err)
        lst_obj.append(obj)

      if save != None:
        tp = sp.get_array_module(x)
        np.save("%s/time.npy" % save, np.cumsum(lst_time))
        tp.save("%s/iter_%03d.npy" % (save, k), Ri(x[:n]))
        if stats:
          np.save("%s/dc.npy" % save, lst_dc)
          np.save("%s/obj.npy" % save, lst_obj)

      pbar.update()
      pbar.refresh()

    pbar.close()
    return Ri(x[:n])

def consensus(num_iters, num_normal, A, b, lst_proxg, lamda,
              lst_g=None, method="cg", save=None, verbose=False):
  r"""Consensus Optimization.

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
    num_normal (Int): Number of A.H * A evaluations.
    A (Linop): Forward linear operator.
    b (Array): Measurement.
    lst_proxg (List of Prox): List of proximal operators.
    lamda (Float): ADMM step size.
    lst_g (List of Functions): List of regularizations. If specified,
                               objective over iterations are saved.
    method (String): Determines the method used to solve for the inner
                     least squares.
                     - "cg": Conjugate gradient.
                     - "pi": Polynomial inversion.
    save (None or String): If specified, path to save iterations, costs and
                           timings.
    verbose (Bool): Print information.

  Returns:
    x (Array): Reconstruction.
  """
  device = sp.get_device(b)
  xp = device.xp

  assert num_normal >= 1
  assert method == "cg" or method == "pi"

  if num_normal == 1 and method == "cg":
    raise Exception("CG requires >= 2 normal evaluations.")

  lst_time = []
  stats = lst_g is not None and save is not None
  if stats:
    lst_obj  = []

  with device:
    l_AHb = lamda * A.H(b)
    x = xp.zeros_like(l_AHb)

    c = len(lst_proxg)
    xi = xp.zeros([1 + c] + list(x.shape), dtype=xp.complex64)
    ui = xp.zeros_like(xi)

    T = sp.linop.Identity(x.shape) + lamda * A.N
    if method == "pi":
      P = optpoly.create_polynomial_preconditioner(num_normal - 1, T, \
                                                   1, 1 + lamda, \
                                                   norm="l_inf", \
                                                   verbose=verbose)

    def prox_f(lamda, v):
      if method == "cg":
        sp.app.App(sp.alg.ConjugateGradient(T, v + l_AHb, v,          \
                                            max_iter=num_normal - 1,  \
                                            tol=1e-4),                \
                                            leave_pbar=False,         \
                                            show_pbar=False,          \
                                            record_time=False).run()
      else:
        v = v - P * (T(v) - (v + l_AHb))
      return v
    
    pbar = tqdm(total=num_iters, desc="Consensus Optimization", \
                leave=True)
    for k in range(num_iters):
      start_time = time.perf_counter()

      xi[0, ...] = prox_f(lamda, x - ui[0, ...])
      for j in range(c):
        xi[1 + j, ...] = lst_proxg[j](lamda, x - ui[1 + j, ...])

      x = xp.mean(xi, axis=0)
      ui += xi - x[None, ...]

      end_time = time.perf_counter()

      dt = end_time - start_time
      lst_time.append(dt)

      if stats:
        obj = sp.to_device(0.5 * xp.linalg.norm(A * x - b)**2, \
                           sp.cpu_device)
        for j in range(c):
          obj += sp.to_device(lst_g[j](x), sp.cpu_device)
        lst_obj.append(obj)

      if save != None:
        tp = sp.get_array_module(x)
        np.save("%s/time.npy" % save, np.cumsum(lst_time))
        tp.save("%s/iter_%03d.npy" % (save, k), x)
        if stats:
          np.save("%s/obj.npy" % save, lst_obj)

      pbar.update()
      pbar.refresh()

    pbar.close()
    return x
