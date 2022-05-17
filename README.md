# Polynomial Preconditioner for Compressed Sensing

This repository reproduces the experiments in
[Accelerating Convergence of Proximal Methods for Compressed Sensing using
Polynomials with Application to MRI](https://arxiv.org/abs/2204.10252).

Videos of results:
  - [3D Subspace MRF](https://raw.githubusercontent.com/sidward/ppcs/main/videos/mrf.mp4)
  - [2D Radial](https://raw.githubusercontent.com/sidward/ppcs/main/videos/radial.mp4)
  - [2D Cartesian](https://raw.githubusercontent.com/sidward/ppcs/main/videos/brain.mp4)
  - [2D Spiral](https://raw.githubusercontent.com/sidward/ppcs/main/videos/spiral.mp4)

Written by Siddharth Srinivasan. Please post an issue on the repository
page if there is a problem.

## Installation.

Run the following commands in sequence to run the experiments.

1. `conda update -n base -c defaults conda`
2. `make conda`
3. `conda activate ppcs`
4. `make pip`

**Troubleshooting**:

1. This repository was tested on an NVIDIA GPU. If running on a system without the same, please remove the following packages from `environment.yaml`:
   - `cudnn`
   - `nccl`
   - `cupy`
2. Additionally, if not using an NVIDIA GPU, please set `devnum = -1` for each of the `demo_*.py` files.
3. When running `make pip`, `git clone git@github.com:mlazaric/Chebyshev.git` will error if GitHub ssh keys are not set. Please replace that line in the `Makefile` with `git clone https://github.com/mlazaric/Chebyshev.git` and run `make pip` again.

## Data.

For most experiments, the corresponding data are located in the `data`
folder.
For the `mrf3d` experiment, please run `bash download_data.sh` in the
`data/mrf3d` folder to download the data.

## Run experiments.

- All experiments are in the form `demo_*.py`.
- An experiment can be performed by running, for example, `python3 demo_brain.py`.
- The respective Jupyter notebooks (as in, `plot_*.ipynb`) can be used to generate images.
- The Jupyter notebooks must be started after running Step 3 above.
- Additionally, LaTeX is required to render equations in the plots.

## Uninstall.

To uninstall, run the following commands:

1. `conda activate`
2. `make clean`

## Packages used:

- [SigPy](https://github.com/mikgroup/sigpy) [![DOI](https://zenodo.org/badge/139635485.svg)](https://zenodo.org/badge/latestdoi/139635485)
- [SymPy](https://github.com/sympy/sympy) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5558034.svg)](https://doi.org/10.5281/zenodo.5558034)
- [Chebyshev](https://github.com/mlazaric/Chebyshev) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5831845.svg)](https://doi.org/10.5281/zenodo.5831845)

## DOI.
[![DOI](https://zenodo.org/badge/452385092.svg)](https://zenodo.org/badge/latestdoi/452385092)
