# Let's build & publish the documentation

> Note: In order to ease the code review, please consider submitting your modifications and build in different commits

In the `substra-documentation` repository:

```sh
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

