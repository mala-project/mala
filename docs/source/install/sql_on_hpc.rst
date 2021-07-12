Using postgres on HPC infrastructure (for optuna)
====================================================

Hemera5 (HZDR)
--------------

1. Install postgres (e.g. in a conda environment)
2. Initialize a postgres server configuration somewhere in your home directory:

   .. code-block:: bash

      $ cd /a/suitable/location/
      $ initdb -D YOUR_SERVER_NAME

3. Edit the configuration files of this server:

    - In :file:`postgres_local/postgresql.conf` add/change the line containing listen_adresses so that it reads:
      ``listen_addresses = '*'``

    - In :file:`pg_hba.conf`, under ``IPv$`` local connections add the line:
      ``host    all             all             0.0.0.0/0               trust``

4. Start the postgres server:

   .. code-block:: bash

      $ pg_ctl -D postgres_local -l logfile start

5. Create the "username database"

    - This is needed in order to access the psql interface (for maintenance): ``createdb``

6. Create a database for your hyperparameter optimizations: ``createdb YOUR_DATABASE_NAME``

7. Host a postgres server via a compute job:

    - You can/should also create a database for optuna, so you can create a study from within Python

      .. code-block:: bash

          cd /a/suitable/location/
          pg_ctl -D A_DESCRIPTIVE_SERVER_NAME -l logfile start
          createdb YOUR_SERVER_NAME

          while true; do
            pg_ctl -D postgres_local status
            sleep 60m
          done

8. MALA/Optuna can now connect to this database via

    .. code-block:: python

        parameters.hyperparameters.rdb_storage = "postgresql://YOUR_USER_NAME@YOUR_COMPUTE_NODE/YOUR_DATABASE_NAME"
        
9. [optional] To increase the maximum number of connections to PostgreSQL (default is 100), go to ``a/suitable/location/postgres_local/postgresql.conf`` and change the value of ``max_connections``. In addition, the value of ``shared_buffers`` is recommended to be set to about 10% of RAM available on a compute node. Note that the actual number of connections exceeds the number of workers, likely because the workers establish new connections every time they communicate with DB and idle connections not dying out by default.
