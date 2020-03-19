# Let's build & publish the documentation

## Commands

```sh
make clean && make html && cp -R _build/html/* docs/ && echo "Wouhou! New build moved to docs!"
git add .
git commit -m "Documentation Update $(date -u +"%Y-%m-%d %H:%M")"
git push
```

## Source Tree

```sh
├── contribute
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
├── overview
│   └── overview.md
├── publish.md
└── README.md
```
