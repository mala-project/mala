import mala

from mala.datahandling.data_repo import data_path
import os

parameters = mala.Parameters()
parameters.descriptors.descriptors_contain_xyz = False

# First, convert from Numpy files to openPMD.

data_converter = mala.DataConverter(parameters)

for snapshot in range(2):
    data_converter.add_snapshot(
        descriptor_input_type="numpy",
        descriptor_input_path=os.path.join(
            data_path, "Be_snapshot{}.in.npy".format(snapshot)
        ),
        target_input_type="numpy",
        target_input_path=os.path.join(
            data_path, "Be_snapshot{}.out.npy".format(snapshot)
        ),
        additional_info_input_type=None,
        additional_info_input_path=None,
        target_units=None,
    )

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="converted_from_numpy_*.bp5",
    descriptor_calculation_kwargs={"working_directory": "./"},
)

# Convert those files back to Numpy to verify the data stays the same.

data_converter = mala.DataConverter(parameters)

for snapshot in range(2):
    data_converter.add_snapshot(
        descriptor_input_type="openpmd",
        descriptor_input_path="converted_from_numpy_{}.in.bp5".format(
            snapshot
        ),
        target_input_type="openpmd",
        target_input_path="converted_from_numpy_{}.out.bp5".format(snapshot),
        additional_info_input_type=None,
        additional_info_input_path=None,
        target_units=None,
    )

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="verify_against_original_numpy_data_*.npy",
    descriptor_calculation_kwargs={"working_directory": "./"},
)

for snapshot in range(2):
    for i_o in ["in", "out"]:
        original = os.path.join(
            data_path, "Be_snapshot{}.{}.npy".format(snapshot, i_o)
        )
        roundtrip = "verify_against_original_numpy_data_{}.{}.npy".format(
            snapshot, i_o
        )
        import numpy as np

        original_a = np.load(original)
        roundtrip_a = np.load(roundtrip)
        np.testing.assert_allclose(original_a, roundtrip_a)

# Now, convert some openPMD data back to Numpy.

data_converter = mala.DataConverter(parameters)

for snapshot in range(2):
    data_converter.add_snapshot(
        descriptor_input_type="openpmd",
        descriptor_input_path=os.path.join(
            data_path, "Be_snapshot{}.in.h5".format(snapshot)
        ),
        target_input_type="openpmd",
        target_input_path=os.path.join(
            data_path, "Be_snapshot{}.out.h5".format(snapshot)
        ),
        additional_info_input_type=None,
        additional_info_input_path=None,
        target_units=None,
    )

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="converted_from_openpmd_*.npy",
    descriptor_calculation_kwargs={"working_directory": "./"},
)
