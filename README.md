# Let's build & publish the documentation

## Repositories

- Substra [repository](https://github.com/SubstraFoundation/substra)
- Substra-documentation [repository](https://github.com/SubstraFoundation/substra-documentation/)
- Substra [documentation](https://doc.substra.ai/)

## Files & Directories

> Note: The `CONTRIBUTING.md` orignal source file is located in the [.github repository](https://github.com/SubstraFoundation/.github) and then fetched into this documentation.

- `_static` contains static files (favicon, css, logo)
- `src` contains all the source documents
- `docs` contains all the built documentation
- `conf.py` is the Python Sphinx configuration file
- `index.rst` is the general Table Of Content of this very documentation
- `_build` is the default build output directory

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

## Source Tree

```sh
├── contribute
│   └── CONTRIBUTING.md
├── entry_points
│   ├── faq.md
│   └── glossary.md
├── getting_started
│   ├── installation
│   │   ├── local_install_docker_compose.md
│   │   └── local_install_skaffold.md
│   └── usage
│       └── usage.md
├── img
│   ├── architecture_overview.png
│   ├── git_pr_status.png
│   ├── start_backend.png
│   ├── start_frontend.png
│   └── start_hlf-k8s.png
├── overview
│   └── overview.md
├── platform_description
│   └── platform.md
└── publish.md
```
