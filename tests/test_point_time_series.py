import os

import numpy as np
import numpy.testing as npt

from pyTSEB.TSEBConfigFileInterface import TSEBConfigFileInterface


TEST_CONFIG = 'tests/Config_PointTimeSeries.txt'
CSV_OUT_PATH = 'tests/data/output/OutputTest.txt'
OLD_CSV_PATH = 'tests/data/OutputTest.txt'


def test_csv_nodiff():
    # Create an interface instance
    setup = TSEBConfigFileInterface()
    # Get the data from configuration file
    config_data = setup.parse_input_config(TEST_CONFIG, is_image=False)
    setup.get_data(config_data, is_image=False)
    # Run the model
    setup.run(is_image=False)

    new_data = np.loadtxt(CSV_OUT_PATH, skiprows=1)
    old_data = np.loadtxt(OLD_CSV_PATH, skiprows=1)

    npt.assert_allclose(new_data, old_data)

    os.remove(CSV_OUT_PATH)
