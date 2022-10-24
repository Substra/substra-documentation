# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Add How to download a model on the substrafl example (#204)
- Update colors & ux (#209)

## [0.22.0]

- Update the examples metrics with the change on Metrics from substratools (#183)
- Update the examples with the generic tasks
- feat: provide hosted jupyter notebook examples (#195)
- Update the examples with the changes on algo to function in substratools (#201)

## [0.21.0]

- fix: Environment variable not set properly in Substrafl example
- fix: Substrafl example not working in docker mode (change torch version)
- chore: change the Dockerfile `ENTRYPOINT` definition of algorithm relying on `substratools` to pass
  the method to execute under the `--method-name` argument
- chore: change the assets and tools algo to feed with inputs outputs dictionary
- feat: rename Connect to Substra
- doc: update GPU page. The data are now automatically move to cpu or gpu by substrafl.
- chore: rename connect-tools to substra-tools
- fix: mnist example plot by using the real organization ID of the clients
- fix: mlflow script
- doc: add documentation on the execution steps in remote mode and the centralized strategy workflow (#173)
- doc: remove download all examples button
- doc: container build error logs are now accessible to end users

## [0.20.0]

- doc: add GPU page on substra documentation
- doc: add new "Deployment" section with some upgrade notes
- doc: remove the `DEBUG_OWNER` mechanism, the substra local clients share the same db and have their own organization_id

## [0.19.0]

- doc: add documentation on the remote backend
- feat: add seed to mnist example
- Drop Python 3.7 support

## [0.18.0]

- fix: mnist example plot
- chore: update mnist example to include torch Dataset class as argument for torch Algo

## [0.17.0]

- fix: mnist example plot
- chore: update mnist example to include torch Dataset class as argument for torch Algo

## [0.16.0]

- fix: new filters api
- doc: update get_performances.rst

## [0.15.0]

- doc: add Performance monitoring in local mode page
- feat: rename node to organization

## [0.14.0]

### Fixes

- fix: broken link in sdk section

### Added

- docs: add compatibility tables
- fix: broken link in sdk section

## [0.13.0]

## [0.12.0]

## [0.11.0]

- feat: all but release and stable versions uses the main branch of the components
- feat: RTD builds the examples

## [0.10.0]

- feat: index clean up
- update substra-tools version in example Dockerfiles
- add clean instruction to Makefile
- fix the substratools reference warning

## [0.9.0]

## [0.8.1]

## [0.8.0]

- feat: Substrafl FedAvg example on MNIST dataset by @ThibaultFy in <https://github.com/Substra/substra-documentation/pull/36>
- feat: Titanic Substra example

## [0.7.0] - 2022-01-19

### Added

- docs: add page -> How to test code and understand errors

### Fixes

- chore: explain what are models and specs
- doc(release): fix links and add links

## [0.6.0] - 2022-01-19

- chore: Substra 0.6.0 - update substra to 0.16.0 by @Esadruhn in <https://github.com/Substra/substra-documentation/pull/23>

## [0.5.0] - 2022-01-12

### Added

- Add look and feel override to RTD theme by @jmorel in <https://github.com/Substra/substra-documentation/pull/11>
- remove unused files by @maikia in <https://github.com/Substra/substra-documentation/pull/15>
- save artefacts only when needed by @maikia in <https://github.com/Substra/substra-documentation/pull/14>
- feat: doc content full rewrite by @RomainGoussault in <https://github.com/Substra/substra-documentation/pull/9>
- Fail the build when there are warnings by @RomainGoussault in <https://github.com/Substra/substra-documentation/pull/16>
- doc: include Substra API Reference by @Fabien-GELUS in <https://github.com/Substra/substra-documentation/pull/17>
- feat: Substra versioning by @Fabien-GELUS in <https://github.com/Substra/substra-documentation/pull/18>
- chore: update substra to 0.15.0 by @Esadruhn in <https://github.com/Substra/substra-documentation/pull/19>

## [0.1.1] - 2021-12-07

### Added

- Skeleton for Substra Documentation

- Examples are build by Sphinx-gallery in CI and copied to artifacts
- Artifacts are copied to read the docs and the rest of documentation is then build
