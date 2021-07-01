Access central storage servers from locations other than Hemera (Linux)

https://fwcc.pages.hzdr.de/infohub/hpc/storage.html
sshfs :username@hemera5.fz-rossendorf.de folder/file/location folder/location/in/local

This is done to mount any folder on the server to the local device. In tensorboard the --logdir which is the directory you will create data to visualize. Files that TensorBoard saves data into are called event files. These files have to be on the local device so as to get access. Hence the mounting is done.
