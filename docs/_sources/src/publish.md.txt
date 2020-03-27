# Let's build & publish the documentation

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
