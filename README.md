# JustCause

Comparing algorithms for causality analysis in a fair and just way.

## Description

A **work in progress** for causal estimator evaluation. The framework aims to make comparison of
methods easier, by allowing to compare them across both generated and existing datasets.

#### ToDos:

* make the package itself independent of Sacred, just advocate it as best practice
* migrate all files ending in `-old` and delete them if no longer necessary
* create some proper unittests and use pytest instead of the shipped unittest
* add the final bachelor thesis as pdf under `references`
* use Sphinx (checkout `docs` folder) to create command reference and some explanations.
* convert `configs/config.py` into a `config.yaml`
  * the content of config should not accessed from everywhere but only the necessary information passed! 
* adhere to `pep8` and other standards. Use `pre-commit` (which is set up below) to check and correct all mistakes  
* Don't fix things like random seed within the package, it's a library, advocate to do this outside (name this best-practice within the docs)
* separate modules that only do math from plotting modules. Why would the generators/acic module need matplotlib as dependency
* follow import order, first Python internal modules, then external, then the modules of your package.
* use PyCharm and check for the curly yellow underline hints how to improve the code
* add some example notebooks in the notebooks folder
* add the libraries which a required (no visualisation) into setup.cfg under requires.
* Check licences of third-party methods and add and note them accordingly. Within the __init__.py of the subpackage add a docstring and state the licences and the original authors. 
* Do not set environment variables inside library, rather state this somewhere in the docs. os.environ['L_ALL'] 
* Never print something in a library, use the logging module for logging. Takes a while to comprehend
* move the `experiment.py` module into the `scripts` folder because it's actually using the package (fix the imports accordingly)
* avoid plotting to `results/plots/S-Learner - LinearRegressionrobustness.png'` in the unittests (right now the directory needs to be created for the unittests to run)
* Do imports within functions only when really necessary (there are rare cases only) otherwise on the top of the module
* Don't set `R_HOME` environment variable and rely on what conda is doing for you. Avoid setting any kind of path via environment variables.
* Remove all `if __name__ == "__main__":` sections from the modules in the justcause package

## Installation

In order to set up the necessary environment:

1. create an environment `justcause` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate justcause
   ```
3. install `justcause` with:
   ```
   python setup.py install # or `develop`
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

5. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yaml` for the exact reproduction of your
   environment with:
   ```
   conda env export -n justcause -f environment.lock.yaml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```
   conda env update -f environment.lock.yaml --prune
   ```
## Project Organization

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── justcause           <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Note

This project has been set up using PyScaffold 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
