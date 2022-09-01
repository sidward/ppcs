.PHONY: conda pip clean

conda:
	conda env create -f environment.yaml

update:
	conda env update --file environment.yaml --prune

pip:
	pip install git+https://github.com/mikgroup/sigpy.git@master
	git clone git@github.com:mlazaric/Chebyshev.git
	pip install imageio-ffmpeg

clean:
	rm -rf Chebyshev
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	conda env remove -n ppcs
