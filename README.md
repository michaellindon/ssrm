![Build](https://github.com/optimizely/ssrm/workflows/Build/badge.svg)

# <img src="logos/ssrm-blue.png" alt="ssrm-logo" width="40"/> SSRM: A Sequential Sample Ratio Mismatch Test
A package for sequential testing of Sample Ratio Mismatch (SRM).

Contributors:
- Michael Lindon (michael.lindon@optimizely.com )

## Installation
We recommend that you use an isolated virtual environment to install and run the code in this repo (See: [virtualenv](https://pypi.org/project/virtualenv/) and [pyenv](https://github.com/pyenv/pyenv))

1. You can clone the repository from the command-line:
    ```console
    git clone git@github.com:optimizely/ssrm.git
    ```
    We highly recommend that you branch off of `master` and pull in upstream changes regularly.
1. Install dependencies: Run `make install`.
    - If you wish to develop in the repo, run `make install-dev`.  Also, see the contributing doc [here](https://github.com/optimizely/ssrm/blob/master/CONTRIBUTING.md)
    > **Tip:** have a look in the [Makefile](https://github.com/optimizely/ssrm/blob/master/Makefile) to learn more about what this, and other make recipes do!
1. Run tests:
    -   `make check` to run all checks.
    -   `make test` to run unit tests.


## Tutorials
We provide a tutorial notebook that walks through an example of running a
Sequential SRM test
[here](https://github.com/optimizely/ssrm/blob/master/notebooks/introduction.ipynb).  Run `jupyter lab`, and open `notebooks/introduction.ipynb`.

## Documentation
The latest reference documentation is [here](https://ssrm.readthedocs.io/en/latest/).

## Contributing
See the contributing doc [here](https://github.com/optimizely/ssrm/blob/master/CONTRIBUTING.md).

### Credits
First-party code (under `ssrm_test`) is copyright Optimizely, Inc. and contributors, licensed under Apache 2.0.

### Additional Code
This software incorporates code from the following open source projects:

**numpy** [https://numpy.org/index.html](https://numpy.org/index.html)
- Copyright © 2005-2020, NumPy Developers.
- License (BSD): https://numpy.org/license.html#license

**scipy** [https://www.scipy.org/scipylib/index.html](https://www.scipy.org/scipylib/index.html)
- Copyright © 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
- License (BSD): https://www.scipy.org/scipylib/license.html

**toolz** [https://github.com/pytoolz/toolz](https://github.com/pytoolz/toolz)
- Copyright © 2013 Matthew Rocklin
- License (New BSD): https://github.com/pytoolz/toolz/blob/master/LICENSE.txt

**typing** [https://github.com/python/typing](https://github.com/python/typing)
- Copyright © 2001-2014 Python Software Foundation; All Rights Reserved.
- License (Python Software Foundation License (PSF)): https://github.com/python/typing/blob/master/LICENSE
