How-to use new API tokens for login
===================================

This short guide explains how to manage API tokens in the web application, and use them in the Substra SDK.

.. admonition:: Why generate API tokens?

   The Substra SDK provides a way to log in using username and password (see `substra.Client <references/sdk.html#client>`_).
   
   It is safe, but should be used with caution, as it:
   
   * doesn't allow for a precise lifetime or separating concerns by creating one token per purpose
   
   * may surprise or limit you through its underlying automated session management
   
   * can encourage using cleartext passwords, which can end up shared in version control.
   
   For these reasons, it is possible for Substra node administrators to disable "implicit login" and force users to generate tokens in the web app.
   
   Whatever the situation, you should use a mechanism to ensure credentials are kept out of view, for instance by reading secret files or environment variables at runtime (see :ref:`client configuration howto`)
   

.. warning::
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