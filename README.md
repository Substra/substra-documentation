[![Documentation
Status](https://readthedocs.com/projects/owkin-connect-documentation/badge/?version=latest&token=184028ad6219084eb2c5dbdacc299817e7cd88cbf48e940b260793e8f48dc591)](https://owkin-connect-documentation.readthedocs-hosted.com/en/latest/?badge=latest)

# Substra documentation

Welcome to Connect documentation. [Here](https://owkin-connect-documentation.readthedocs-hosted.com/en/latest/index.html) you can find the latest version.
## Contributing

If you would like to contribute to this documentation please clone it locally and make a new branch with the suggested changes.

To deploy the documentation locally you will need to install all the necessary requirements which you can find in the 'requirements.txt' file of the root of this repository. You can use pip in your terminal to install it: `pip install -r requirements.txt`

Install substra: `pip install substra`
Next, to build the documentation move to the docs directory: `cd docs`

And then: `make html`

The first time you run it or if you updated the examples library it may take a little longer to build the whole documentation.

To see the doc on your browser : `make live-html`
And then go to http://127.0.0.1:8000

Once you are happy with your changes push your branch and make a pull request.

Thank you for helping us improving!

## To publish on RTD before merging your PR:

```sh
    # See the tags
    git tag
    # Add a new tag
    git tag -a X.X.X-your-branch
    git push origin X.X.X-your-branch
```

Check the publish action is running: https://github.com/owkin/connect-documentation/actions

Activate your version on RTD (need admin rights): https://readthedocs.com/projects/owkin-connect-documentation/versions/

Follow the build here : https://readthedocs.com/projects/owkin-connect-documentation/builds/

See the doc on https://owkin-connect-documentation.readthedocs-hosted.com/en/X.X.X-your-branch

If everything is OK, you can delete your version on RTD (wipe button): https://readthedocs.com/projects/owkin-connect-documentation/versions/
and delete your tag : `git push --delete origin X.X.X-your-branch`
