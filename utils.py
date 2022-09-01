import os
import numpy as np
import sigpy as sp
from numpy.polynomial import Polynomial, Chebyshev

def normal_helper(max_iter, n_normal):
  lst = [k for k in range(2, max_iter + 1) if max_iter % k == 0]
  return lst[:n_normal]

def lamda_helper(lamda):

  string = ("%e" % lamda).split("e")
  significand = np.floor(float(string[0]) * 1e2)/1e2
  exponent = int(string[-1])

  return (significand * 10**exponent, significand, exponent)


def quality_filter(base, experiment, eps, verbose=True):

  params_file = "results/%s/%s_params.npy" % (base, experiment)
  params = np.load(params_file, allow_pickle=True).item()

  if "lst_normal" in params:
    lst_params = []
    for elm in params["lst_lamda"]:
      for normal in params["lst_normal"]:
        lst_params.append(tuple(list(elm[1:]) + [normal]))
  else:
    lst_params = [tuple(elm[1:]) for elm in params["lst_lamda"]]

  (lst_filtered, lst_err) = ([], [])
  for elm in lst_params:

    case = ("%3.2fx10^%d"    % elm) if len(elm) == 2 else \
           ("%3.2fx10^%d_%d" % elm)

    # Load times.
    e = np.load("results/%s/%s_%s/err.npy" % (base, experiment, case))[1, -1]
    if eps is None or e < eps:
      lst_err.append(e)
      lst_filtered.append(elm) 

  if eps is not None:
    if verbose:
      print(f"Selected {len(lst_filtered)} of {len(lst_params)}.")
    return (lst_filtered, lst_err)

  if verbose:
    print(f"Selected best reconstruction parameters.")
  return lst_filtered[np.argmin(lst_err)]


def calc_conv(base, ord=2, device=0):
  if os.path.isfile(f"{base}/conv.npy"):
    return np.load(f"{base}/conv.npy", mmap_mode="r")

  dev = sp.Device(device)
  xp = dev.xp
  with dev:
    time = np.load(f"{base}/time.npy")
    target = xp.load(f"{base}/iter_%03d.npy" % (time.size - 1))
    scale  = xp.linalg.norm(target.ravel(), ord=ord)

    conv = []
    for k in range(time.size):
      dif = xp.load(f"{base}/iter_%03d.npy" % k) - target
      val = xp.linalg.norm(dif.ravel(), ord=ord)/scale
      del dif
      conv.append(val)
    conv = sp.to_device(100 * xp.array(conv), sp.cpu_device)

  np.save(f"{base}/conv.npy", conv)
  return conv


def get_best_case(base, experiment, eps, verbose=True):
  loc = f"results/{base}/{experiment}"
  (params, _) = quality_filter(base, experiment, eps=eps, verbose=verbose)

  assert len(params) > 0

  (lst_cost, lst_aha, lst_conv, lst_time) = ([], [], [], [])
  for elm in params:

    if len(elm) == 2:
      src = loc + "_%3.2fx10^%d/" % elm
      num_normal = 1
    else:
      src = loc + "_%3.2fx10^%d_%d/" % elm
      num_normal = elm[-1]
    
    conv = calc_conv(src)
    time = np.load(f"{src}/time.npy")
    err  = np.load(f"{src}/err.npy")
    aha  = np.arange(conv.size) * num_normal

    c = Chebyshev.fit(time[:-1], np.log(conv[:-1]), deg=1)
    c = c.convert(kind=Polynomial).coef

    lst_cost.append(c[-1])
    lst_aha.append(aha)
    lst_conv.append(conv)
    lst_time.append(time)

    if verbose:
      print(f"CASE: {src.split('/')[-2]} NRMSE: {err[1, -1]:.2f}%")

  idx = np.argmin(lst_cost)
  return (params[idx], lst_aha[idx], lst_conv[idx], lst_time[idx])
