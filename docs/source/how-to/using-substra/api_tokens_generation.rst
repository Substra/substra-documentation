How-to use new API tokens for login
===================================

This short guide explains how to manage API tokens in the web application, and use them in the Substra SDK.

.. note::
   API tokens are node-specific: if your script connects to multiple nodes, generate a token for each of them.

Generating new API tokens
-------------------------

To do so you need to go to the API tokens management page on the web application following this link ``<your-org-name.domain>/manage_tokens``. 
You will see a list of your current tokens as well as an option to generate new ones. 

You can also navigate to the page using the user menu:


.. image:: /documentation/images/find_token_management_page.png


Clicking ``Generate new`` opens a menu allowing you to pick a name and an expiration date for
your new token. 


.. image:: /documentation/images/generate_new_token.png


Afterward your token will be shown only once. Do copy it somewhere safe before proceeding with your work. 


.. image:: /documentation/images/copy_token.png

Using API tokens
----------------

Pass tokens to the `substra.Client <references/sdk.html#client>`_ constructor:

.. code-block:: Python
    :caption: Example of client configuration in code

    client_1 = substra.Client(
        backend_type="remote",
        url="https://org-1.com",
        token="dad943c684f65633635f005b2522a6452d20",
    )

See :ref:`client configuration howto` for other options.

Deleting API tokens
-------------------

Tokens can be deleted using the web application. Be careful, token deletion is irreversible.

If you have scripts using a deleted token, they will no longer execute.