# Let's build & publish the documentation

In the substra-documentation [repository](https://github.com/SubstraFoundation/substra-documentation):

```sh
make clean
make html
touch _build/html/.nojekyll
diff -x "CNAME" -qr _build/html docs
```
