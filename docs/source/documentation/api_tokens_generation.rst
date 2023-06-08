How-to generate new API tokens for login
========================================

This short guide explains how to use the API tokens management page in the web application.

Generating new API tokens
-------------------------

To do so you need to go to the API tokens management page on the web application following this link ``<your-org-name.domain>/manage_tokens``. 
You will see a list of your current tokens as well as an option to generate new ones. 

You can also navigate to the page using the user menu:


.. image:: /documentation/images/find_token_management_page.png


Clicking on the ``Generate new``` button opens a menu allowing you to pick a name and an expiration date for
your new token. 


.. image:: /documentation/images/generate_new_token.png


Afterward your token will be shown only once. Do copy it somewhere safe before proceeding with your work. 


.. image:: /documentation/images/copy_token.png


Deleting API tokens
-------------------

Every token can be deleted using the web application. Do be careful, token deletion is irreversible.
If you have scripts using this deleted token, they will no longer execute. 