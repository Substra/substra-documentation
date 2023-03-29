******************************
Set up SSO with OpenID Connect
******************************

Substra supports external user management through OpenID Connect (OIDC). It is done per-backend, so each organization can use their own user provider and cohabit on the same network.

OIDC users cannot be created as admins (you'll still need to create admin accounts as normal, through ``addAccountOperator.users``), and will all be assigned to a single channel.


Set up the provider
===================

First, you'll need to set up an OIDC client at an identity provider (IP -- or OpenID provider, OP).

The only claims substra need are ``openid email``, which any provider should be able to support. Allow ``<BACKEND URL>/oidc/callback`` as a redirect URI.

Get your **provider URL**. Appending ``/.well-known/openid-configuration`` to this URL should return a JSON description of the provider's capabilities, which Substra will use for much of the configuration (otherwise, you can set endpoints by hand under ``oidc.provider.endpoints``.

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

When a user first signs on through OIDC, they are assigned a username and a channel. 

You must select a channel to which the OIDC users will be assigned with ``oidc.users.channel``.

The username is based on user email, but can be customized with the ``oidc.users.appendDomain`` flag.

A note on user validity
-----------------------

An issue is if a user gets their account disabled at the provider, but still has access to your Substra instance. You can use ``oidc.users.loginValidityDuration`` to mitigate this: accounts that have not logged in in this amount of time are disabled until the user signs on again. The API tokens associated with their account stop working as well, but will work again when they refresh their login.

Since this is irritating to users, Substra will attempt to fetch user info in the background if possible, so the process is transparent to users. However this requires the provider to support offline access and refresh tokens -- not all do, and implementations vary.

Automated login refresh is enabled by default through the setting ``oidc.users.useRefreshToken``, but Substra will disable it and fall back to the manual mode if it can't detect provider support.

If you are using automated login refresh, you can set ``oidc.users.loginValidityDuration`` to a low value to slightly increase security at a small cost in server load. Otherwise, it is a balance of security versus user convenience.


Other settings
==============

If OIDC users will be using the Substra API (for instance if they are data scientists running Pythong scripts), they'll need to generate API tokens on the web frontend and use those in their scripts.

Having to generate new tokens all the time is a hindrance for the users, so you can increase their lifetime through ``config.EXPIRY_TOKEN_LIFETIME`` in the backend values.


Putting it all together
=======================

A minimal working configuration in the backend values would be:

.. code-block:: yaml

   config:
     EXPIRY_TOKEN_LIFETIME: "10080" # one week, in minutes
   oidc:
     enabled: true
     clientSecretName: oidc-secret
     provider:
       url: "PROVIDER_URL"
       displayName: Anything you like here :) It will be displayed to users.
     users:
       channel: CHANNEL_ID
