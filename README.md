# Let's build & publish the documentation

```sh
clean && make html
cp -R _build/html/* docs/
git add . 
git commit -m "Documentation Update"
git push
```