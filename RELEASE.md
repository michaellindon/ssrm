# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.4] - 2020-05-27

### Added
- Configuration for generating documentation on readthedocs.io

## [0.1.3] - 2020-05-26

### Added
- Input data validation

### Fixed
- Running minimum of sequential p values now initialized at 1 so that the correct
range is obtained.

## [0.1.2] - 2020-05-15

### Added
- Jupyter notebook explaining usage of `ssrm_test`.
- LICENSE.txt containing [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0.txt).
- Copyright notices.

### Changed
- Function `sequential_p_values` signature changed to accept data and null
probabilities, instead of Bayes factor.
- Function `sequential_posterior_probabilities` signature changed to accept data and null
probabilities, instead of Bayes factor.

## [0.1.1] - 2020-05-13

### Changed

- Updated Makefile to enable publishing to PyPi.

## [0.1.0] - 2020-05-10

### Added

- Initial release of the Sequential Sample Ratio Mismatch library.
