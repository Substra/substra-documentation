# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.31.0]

- Add user management doc ([#345](https://github.com/Substra/substra-documentation/pull/345))
- Add link to python libraries documentation in Components page ([#347](https://github.com/Substra/substra-documentation/pull/347))
- Add more orchestrator documentation ([#346](https://github.com/Substra/substra-documentation/pull/346))
- Deactivate Binder ([#340](https://github.com/Substra/substra-documentation/pull/340))
- Reorganise documentation according to [diataxis](https://diataxis.fr/) approach ([#330](https://github.com/Substra/substra-documentation/pull/330))
- Added section about channels in the main concepts ([#344](https://github.com/Substra/substra-documentation/pull/344))
- Add frontend documentation in components section ([#346](https://github.com/Substra/substra-documentation/pull/346))
- New page added on Substra Privacy Strategy based on research by Privacy Task Force at Owkin ([#354](https://github.com/Substra/substra-documentation/pull/354))
- Revamp landing page ([#353](https://github.com/Substra/substra-documentation/pull/353))

## [0.30.0]

- New example on how to use the ComputePlanBuilder SubstraFL to compute Federated Analytics on the Sklearn diabetes dataset. ([#311](https://github.com/Substra/substra-documentation/pull/311))
- New example on how to implement a custom cyclic `Strategy` with a `TorchBaseAlgo` with SubstraFL. ([#326](https://github.com/Substra/substra-documentation/pull/326))
- Use `Client.wait_compute_plan` in `substrafl_examples/get_started/run_mnist_torch.py` ([#327](https://github.com/Substra/substra-documentation/pull/327))

## [0.29.0]

- Improve permissions page ([#322](https://github.com/Substra/substra-documentation/pull/322))
- add `shared state` and `local state` definition in SubstraFL overview ([#321](https://github.com/Substra/substra-documentation/pull/321))
- add `rank` definition in the Substra concepts ([#321](https://github.com/Substra/substra-documentation/pull/321))
- Add experiment name for SubstraFL example ([#323](https://github.com/Substra/substra-documentation/pull/323))

## [0.28.0]

- add token management page guide ([#312](https://github.com/Substra/substra-documentation/pull/312))
- Update Iris example with the changes implied by [SubstraFL #120](https://github.com/Substra/substrafl/pull/120) compute plan builder ([#313](https://github.com/Substra/substra-documentation/pull/313))
- Add caution for frontend cookies when several backends in local ([#308](https://github.com/Substra/substra-documentation/pull/308))
- Update Substra examples to use new `get_task_output_asset` Substra function ([#317](https://github.com/Substra/substra-documentation/pull/317))

## [0.27.0]

- Update the metric registration applying the new [SubstraFL #117](https://github.com/Substra/substrafl/pull/117) feature ([#306](https://github.com/Substra/substra-documentation/pull/306/))
- Use the new way of configuring Substra clients in tutorials ([#305](https://github.com/Substra/substra-documentation/pull/305))
- Add a how-to guide on Substra clients configuration ([#307](https://github.com/Substra/substra-documentation/pull/307))

## [0.26.3]

- Rename function names in examples ([#302](https://github.com/Substra/substra-documentation/pull/302))

## [0.26.2]

- Fix Binder build on tags ([#301](https://github.com/Substra/substra-documentation/pull/301))
- Add hardware requirements ([#300](https://github.com/Substra/substra-documentation/pull/300))

## [0.26.0]

- Update Iris example to init model parameter in the new initialisation task ([#289](https://github.com/Substra/substra-documentation/pull/289))
- `Algo`as `Strategy` parameter in SubstraFL examples (#287)
- Clarify how to login in the remote mode (#281)
- Improve permission page (#279)
- Improve installation paragraph in landing page (#276)
- Rename substra SDK Algo to Function (#264)
- Apply EvaluationStrategy modifications of [SubstraFL #85](https://github.com/Substra/substrafl/pull/85) (#273)

## [0.25.0]

- Add contributing guide & code of conduct md files (#253)
- Remove test only for datasamples in examples (#246)
- Rename examples from `plot_*.py` to `run_*.py` (#268)
- Add note on supported OS (#270)

## [0.24.0]

- Update intro scheme and index generator scheme (#245)
- Add components scheme (#245)
- Fix order of build in yaml file (#241)
- Add upgrade note for 0.23.1 (#228)
- Add contributing guide & code of conduct pages to Substra doc (#230)

## [0.23.1]

- Add Mnist example image (#218)
- Add a Sickit-learn example for SubstraFL to the documentation (#206)
- Update the Susbtra sdk doc to remove references to tuples (composite, aggregate...) or train and predict
  concepts (#215)
- Add How to download a model on the substrafl example (#204)
- Update colors & ux (#209)
- Update the MNIST example to be easier to read (#208)
- Move utils folder content in assets folder (#210)
- Increase tocdepth in SDK API reference to display methods under Client (#219)
- Update substratools Docker image name (#222)
- Use the new `add_metric` SubstraFL function in examples (#214)
- Add documentation on mTLS setup (#200)

## [0.22.0]

- Update the examples metrics with the change on Metrics from substratools (#183)
- Update the examples with the generic tasks
- feat: provide hosted jupyter notebook examples (#195)
- Update the examples with the changes on algo to function in substratools (#201)
- Register functions in substratools using decorator `@tools.register` (#202)

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
