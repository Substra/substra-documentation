Upgrade notes
=============

Connect 0.15.0
--------------

This version ships with an upgrade to dependencies including `an upgrade to PostgreSQL <https://docs.bitnami.com/kubernetes/infrastructure/postgresql/administration/upgrade/#to-1100>`__. You need to upgrade the database **before** upgrading the application proper, to ensure migrations work properly.

Postgres is used in the orchestrator and also in the backend (for substrapp).

The most straightforward way to upgrade is to:

- dump the database
    - for substrapp, it's easiest to `do so through Django <https://github.com/owkin/connect-backend/blob/016806fc8e43da4d566425cfbae9c73d5256337e/UPGRADE.md#backup-and-restore-django-databases>`__ (you get a JSON file)
    - for the orchestrator, connect to the database and run ``pg_dumpall`` (you get an SQL file)
    - don't pay much attention to localrep, it will be rebuilt
- destroy the Postgres Kubernetes objects
- determine what is the smallest chart version that gets you the new dependencies
    - it should be orchestrator ``7.0.0`` and backend ``18.0.0``
    - adjust your values structure to match changes (only minimal structure changes, don't update docker tags)
- run ``helm upgrade``
- restore the database, either through Django (substrapp) or by running your SQL file on the database (orchestrator)
- actually run the upgrade to Connect 0.15.0 proper
