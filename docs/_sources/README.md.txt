# Let's build & publish the documentation

## Commands

```sh
# In the substra-documentation repository:
make clean && make html && cp -R _build/html/* docs/ && echo "Wouhou! New build moved to /docs"
git add .
git commit -m "Documentation Update $(date -u +"%Y-%m-%d %H:%M")"
git push

# Copy the build to the github page:
cp -R docs/* ../SubstraFoundation.github.io/
cd ../SubstraFoundation.github.io/
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
│   └── git_pr_status.png
├── overview
│   └── overview.md
├── platform_description
│   └── platform.md
└── publish.md
```
