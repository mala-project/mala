# Set up tensorboard to visualize data from HPC cluster

## To mount Hemera on to local device you can find link below

https://fwcc.pages.hzdr.de/infohub/hpc/storage.html

or
## Use the following command

sshfs username@hemera5.fz-rossendorf.de:folder/file/location folder/location/in/local

This is done to mount any folder on the server to the local device. In tensorboard the --logdir which is the directory you will create data to visualize. Files that TensorBoard saves data into are called event files. These files have to be on the local device so as to get access. Hence the mounting is done.
