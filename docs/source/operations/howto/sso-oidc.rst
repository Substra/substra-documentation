******************************
Set up SSO with OpenID Connect
******************************

Substra supports external user management through OpenID Connect (OIDC). It is done per-backend, so each organization can use their own user provider and cohabit on the same network.

OIDC users cannot be created as admins: you'll still need to create at least one admin account as normal, through ``addAccountOperator.users``. OIDC users will all be assigned to a single channel.


Set up the provider
===================

First, set up an OIDC client at an identity provider (IP -- or OpenID provider, OP).

The only claims Substra needs are ``openid email``, which any provider should be able to support. Allow ``<BACKEND URL>/oidc/callback`` as a redirect URI.

Get your **provider URL**. Appending ``/.well-known/openid-configuration`` to this URL should return a JSON description of the provider's capabilities, which Substra will use for much of the configuration. Otherwise, you can set endpoints by hand under ``oidc.provider.endpoints``.

The provider will give you a **client id** and a **client secret**. Deploy them on the cluster in a secret:

.. code-block:: yaml

   apiVersion: v1
   kind: Secret
   metadata:
     name: oidc-secret
   stringData:
     OIDC_RP_CLIENT_ID: "CLIENT_ID"
     OIDC_RP_CLIENT_SECRET: "CLIENT_SECRET"


Set up user creation
====================

When a user first logs in through OIDC, they are assigned a username based on their email address. The ``oidc.users.appendDomain`` flag controls whether email domain is included.

You must choose one user creation process:
* Set up a default channel by setting ``oidc.users.channel`` to the name of an existing channel (see the value of ``orchestrator.channels``). OIDC users will be able to use the platform right away.
* Alternatively, set ``oidc.users.requireApproval`` to ``true``: after their first login, OIDC users will have to wait for manual approval from an administrator (on the web frontend).

.. admonition:: Note on user validity

   Substra OIDC users accounts will remain valid for a bit after the correspond account at the provider has been disabled; this can be an issue if, for instance, an employee has been recently terminated but still has access to the Substra instance.
   
   This can be mitigated through ``oidc.users.loginValidityDuration``: accounts that have not logged in in this amount of time (seconds) are disabled until the user logs in again. The API tokens associated with their account stop working as well, but will work again when they refresh their login.
   
   To avoid irritating users with frequent login prompts, Substra will attempt to do this in the background, making all this invisible to users. However this requires the provider to support offline access and refresh tokens -- not all do, and implementations vary.
   
   Automated login refresh is enabled by default through the setting ``oidc.users.useRefreshToken``, but Substra will disable it and fall back to the manual mode (actual login prompts) if it can't detect provider support.
   
   If you are using automated login refresh, you can set ``oidc.users.loginValidityDuration`` to a low value to slightly increase security at a small cost in server load. Otherwise, it is a balance of security versus user convenience.


Other settings
==============

If OIDC users will be using the Substra API (for instance if they are data scientists running Python scripts), they'll need to generate API tokens on the web frontend and use those in their scripts.

Having to generate new tokens all the time is a hindrance for the users: you can increase their lifetime through ``config.EXPIRY_TOKEN_LIFETIME`` in the backend values.


Putting it all together
=======================

Example of a minimal working configuration in the backend values:

.. code-block:: yaml

   config:
     EXPIRY_TOKEN_LIFETIME: "10080" # one week, in minutes
   oidc:
     enabled: true
     clientSecretName: oidc-secret # set earlier
     provider:
       url: "PROVIDER_URL"
       displayName: "PROVIDER_NAME" # will be displayed on the login page
     users:
       channel: "CHANNEL_ID"
