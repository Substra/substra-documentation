# Let's build & publish the documentation

## Files & Directories

> Note: The `CONTRIBUTING.md` orignal source file is located in the [.github repository](https://github.com/SubstraFoundation/.github) and then fetched into this documentation.

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
git commit -m "Documentation Update $(date -u +"%Y-%m-%d %H:%M")"
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
