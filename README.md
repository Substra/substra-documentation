# Let's build & publish the documentation

[![Substra Repository](https://img.shields.io/badge/substra-repo-brightgreen)](https://github.com/SubstraFoundation/substra)
[![Substra Documentation Repository](https://img.shields.io/badge/doc-repo-brightgreen)](https://github.com/SubstraFoundation/substra-documentation/)
[![Build Status](https://travis-ci.org/SubstraFoundation/substra-documentation.svg?branch=master)](https://travis-ci.org/SubstraFoundation/substra-documentation)
[![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-blue)](https://substra.us18.list-manage.com/track/click?e=2effed55c9&id=fa49875322&u=385fa3f9736ea94a1fcca969f)
[![License Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-orange)](https://www.apache.org/licenses/LICENSE-2.0.html)

## Files & Directories

> Note: The `CONTRIBUTING.md` original source file is located in the [.github repository](https://github.com/SubstraFoundation/.github) and then fetched into this documentation.

- `_static` contains static files (favicon, css, logo)
- `src` contains all the source documents
- `docs` contains all the built documentation
- `conf.py` is the Python Sphinx configuration file
- `index.rst` is the general Table Of Content of this very documentation
- `_build` is the default build output directory

## Tree of the `src` folder

```sh
src/
├── architecture.md
├── concepts.md
├── CONTRIBUTING.md
├── debugging.md
├── faq.md
├── glossary.md
├── img
│   ├── architecture_overview.png
│   ├── assets_relationships.png
│   ├── dataset-files-opener.png
│   ├── git_pr_status.png
│   ├── start_backend.png
│   ├── start_frontend.png
│   ├── start_hlf-k8s.png
│   ├── training_phase1.png
│   └── training.svg
├── index.rst
├── overview.md
├── publish.md
├── setup
│   ├── demo.md
│   ├── further_resources.md
│   ├── index.rst
│   └── local_install_skaffold.md
└── usage.md
```

## Commands

```sh
# In the substra-documentation repository:

# Install the dependencies
pip install -r requirements.txt

# Automatically build the documentation at each change and test the result in your browser at http://localhost:8000
make livehtml

# Build the documentation
make docs

# Commit changes
git add .
git commit -m "[build]"
git push
```

## Contributing

- Source code: <https://github.com/SubstraFoundation/substra>
- Issue tracker: <https://github.com/SubstraFoundation/substra/issues>
- Documentation: <https://doc.substra.ai>
- Documentation issue tracker: <https://github.com/SubstraFoundation/substra-documentation/issues>
- Website: <https://www.substra.ai>
- Slack: <https://substra-workspace.slack.com>
- Contribution guidelines: <https://doc.substra.ai/contribute/CONTRIBUTING.html>
- Licence: Apache License 2.0
