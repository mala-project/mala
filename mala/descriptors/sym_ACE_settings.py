# store CG and wigner files in fitsnap3lib/lib/sym_ACE/lib:
from mala.descriptors import gen_labels

topfile = gen_labels.__file__
top_dir = topfile.split("/gen")[0]
lib_path = "%s" % top_dir
