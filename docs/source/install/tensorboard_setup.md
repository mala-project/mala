# Set up tensorboard to visualize data from HPC cluster

Training of surrogate models with MALA is often done on HPC infrastructure, yet the visualization should take place locally. Files that TensorBoard saves data into are called event files. These files have to be on the local device to get access. The best way to achieve this is by either downloading or mounting the relevant folders on the HPC cluster to your local machine.

## Hemera5 (HZDR)

- You can find information on how to mount Hemera onto a local device here:

  <https://fwcc.pages.hzdr.de/infohub/hpc/storage.html>

- Alternatively, simply use the following command

  ```sh
  $ sshfs username@hemera5.fz-rossendorf.de:folder/file/location folder/location/in/local
  ```
