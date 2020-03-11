# Let's build & publish the documentation

```sh
make clean && make html
cp -R _build/html/* docs/
git add . 
git commit -m "Documentation Update $(date -u +"%Y-%m-%d %H:%M")"
git push
```