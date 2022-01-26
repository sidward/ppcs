import numpy as np
import sigpy as sp

def _llr(x, block, lamda, randshift=False, use_eig=False):

  device = sp.get_device(x)
  xp     = device.xp
  eps    = 1e-12
    
  with device:
    (tk, sz, sy, sx) = x.shape
               
    B = sp.linop.ArrayToBlocks((tk, sz, sy, sx),      \
                               (block, block, block), \
                               (block, block, block))
    T = sp.linop.Transpose(B.oshape, (1, 2, 3, 0, 4, 5, 6))
    n = T.oshape[0] * T.oshape[1] * T.oshape[2]
    R = sp.linop.Reshape((n, tk, block**3), T.oshape)
    L = R * T * B

    if randshift:
      shift = [np.random.randint(block) - int(block/2), \
               np.random.randint(block) - int(block/2), \
               np.random.randint(block) - int(block/2)]
      x = xp.roll(x, shift, axis=(1, 2, 3))
    
    mats = L(x)

    if use_eig:
      aha = xp.transpose(mats.conj(), (0, 2, 1)) @ mats
      (v, d, vh) = xp.linalg.svd(aha, full_matrices=False, hermitian=True)

      d = xp.real(d)
      d[d < eps] = 0

      s = xp.sqrt(d).real
      s[xp.isnan(s)] = 0
      s[xp.isinf(s)] = 0

      inv_s = xp.zeros(s.shape, dtype=s.dtype)
      inv_s[s > eps] = 1/s[s > eps]
      u = (mats @ v) * inv_s[:, None, :]

    else:
      (u, s, vh) = xp.linalg.svd(mats, full_matrices=False)

    thresh_s = s - lamda
    thresh_s[thresh_s < 0] = 0

    mats[...] = xp.matmul(u * thresh_s[..., None, :], vh)

    x = L.H(mats)

    if randshift:
      x = xp.roll(x, [-k for k in shift], axis=(1, 2, 3))

    return x

class ProxLLR(sp.prox.Prox):
  def __init__(self, shape, lamda, block, verbose=False):
    self.lamda  = lamda
    self.block  = block
    if verbose:
      print("3D LLR:")
      print("> Lambda: %0.2e" % lamda)
      print("> Block:  %d"    % block)
    super().__init__(shape)

  def _prox(self, alpha, input):
    return _llr(input, self.block, self.lamda * alpha, randshift=True)
