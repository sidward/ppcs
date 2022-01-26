import numpy as np
import sigpy as sp

class ProxTV(sp.prox.Prox):
  def __init__(self, shape, lamda):
    self.lamda  = lamda
    self.G = sp.linop.FiniteDifference(shape)
    self.I = sp.linop.Identity(shape)
    super().__init__(shape)

  def _prox(self, alpha, input):
    dev = sp.get_device(input)
    xp = dev.xp
    with dev:
      T = self.I + (alpha * self.lamda) * self.G.N
      x = input.copy()
      sp.app.App(sp.alg.ConjugateGradient(T, input, x=x,          \
                                          max_iter=100, tol=1e-4),\
                                          leave_pbar=False,       \
                                          show_pbar=False,        \
                                          record_time=False).run()
      return x
