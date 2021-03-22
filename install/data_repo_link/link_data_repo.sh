#!/bin/bash

# I rarely do bash scripting, so feel free to refine this script.

echo "Linking MALA and MALA data repo."
# Get the paths we need for setup.
script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mala_base_path=$( echo ${script_path%/*} )
mala_base_path=$( echo ${mala_base_path%/*} )
examples_path=$mala_base_path/examples
test_path=$mala_base_path/test
pythonfile=data_repo_path.py

# Ask the user for the path to the repo if none was given in the command line.
if [ "$1" != "" ]
then
  data_repo_path=$1
else
  echo "Please input the full path to the MALA data repo."
  read data_repo_path
fi

# Append a / if we have to.
lastcharacter="${data_repo_path: -1}"
if [ "$lastcharacter" != "/" ]
then
  data_repo_path=$data_repo_path/
fi

# Write the python file.
rm -f ${script_path}/${pythonfile}
touch ${script_path}/${pythonfile}
echo "data_repo_path = \"${data_repo_path}\""   >> ${script_path}/${pythonfile}
echo "" >> ${script_path}/${pythonfile}
echo "" >> ${script_path}/${pythonfile}
echo "def get_data_repo_path():"   >> ${script_path}/${pythonfile}
echo "    return data_repo_path"   >> ${script_path}/${pythonfile}

# copy the file to test and example folders.
cp ${script_path}/${pythonfile} ${test_path}
cp ${script_path}/${pythonfile} ${examples_path}
echo "Linking done!"
