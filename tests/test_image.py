import os

import rasterio
import numpy.testing as npt

from pyTSEB.TSEBConfigFileInterface import TSEBConfigFileInterface


TEST_CONFIG = 'tests/Config_LocalImage.txt'
IMG_OUT_PATH = 'tests/data/output/test_image.tif'
ANC_IMG_OUT_PATH = 'tests/data/output/test_image_ancillary.tif'
OLD_IMG_PATH = 'tests/data/test_image.tif'
OLD_ANC_IMG_PATH = 'tests/data/test_image_ancillary.tif'


def test_image_nodiff():
    setup = TSEBConfigFileInterface()
    # Get the data from configuration file
    config_data = setup.parse_input_config(TEST_CONFIG, is_image=True)
    setup.get_data(config_data, is_image=True)
    # Run the model
    setup.run(is_image=True)

    with rasterio.open(IMG_OUT_PATH) as src:
        new_img = src.read()
    with rasterio.open(OLD_IMG_PATH) as src:
        old_img = src.read()
    npt.assert_allclose(new_img, old_img, rtol=1e-4)

    with rasterio.open(ANC_IMG_OUT_PATH) as src:
        new_img = src.read()
    with rasterio.open(OLD_ANC_IMG_PATH) as src:
        old_img = src.read()
    npt.assert_allclose(new_img, old_img, rtol=1e-4)

    os.remove(IMG_OUT_PATH)
    os.remove(ANC_IMG_OUT_PATH)
