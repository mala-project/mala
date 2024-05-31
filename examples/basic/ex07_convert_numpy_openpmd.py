import mala

parameters = mala.Parameters()
data_converter = mala.DataConverter(parameters)

for snapshot in range(2):
    data_converter.add_snapshot(
        descriptor_input_type="openpmd",
        descriptor_input_path="Be_shuffled{}.in.bp4".format(snapshot),
        target_input_type='openpmd',
        target_input_path="Be_shuffled{}.out.bp4".format(snapshot),
        additional_info_input_type=None,
        additional_info_input_path=None,
        target_units=None,
    )

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="Be_snapshot*.bp4",
    descriptor_calculation_kwargs={"working_directory": "./"},
)

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="Be_snapshot*.npy",
    descriptor_calculation_kwargs={"working_directory": "./"},
)

data_converter = mala.DataConverter(parameters)

for snapshot in range(2):
    data_converter.add_snapshot(
        descriptor_input_type="numpy",
        descriptor_input_path="Be_snapshot{}.in.npy".format(snapshot),
        target_input_type='numpy',
        target_input_path="Be_snapshot{}.out.npy".format(snapshot),
        additional_info_input_type=None,
        additional_info_input_path=None,
        target_units=None,
    )

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="Be_snapshot_from_numpy*.bp4",
    descriptor_calculation_kwargs={"working_directory": "./"},
)

data_converter.convert_snapshots(
    descriptor_save_path="./",
    target_save_path="./",
    additional_info_save_path="./",
    naming_scheme="Be_snapshot_from_numpy*.npy",
    descriptor_calculation_kwargs={"working_directory": "./"},
)
