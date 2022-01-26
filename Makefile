.PHONY: conda pip clean

conda:
	conda env create -f environment.yaml

pip:
	pip install git+https://github.com/mikgroup/sigpy.git@master
	git clone git@github.com:mlazaric/Chebyshev.git

clean:
	rm -rf Chebyshev
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	conda env remove -n ppcs
