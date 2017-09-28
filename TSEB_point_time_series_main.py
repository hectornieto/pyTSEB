#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script to run TSEB over a timeseries of point measurements

Created on Dec 29 2015
@author: Hector Nieto

Modified on Jan 13 2016
@author: Hector Nieto

"""

import sys

from pyTSEB.TSEBConfigFileInterface import TSEBConfigFileInterface

config_file = 'Config_PointTimeSeries.txt'


def run_TSEB_from_config_file(config_file):
    # Create an interface instance
    setup = TSEBConfigFileInterface()
    # Get the data from configuration file
    config_data = setup.parse_input_config(config_file, is_image=False)
    setup.get_data(config_data, is_image=False)
    # Run the model
    in_data, out_data = setup.run(is_image=False)

    # Can analyse/visualise the in_data and out_data pandas DataFrames here
    # ax = out_data.loc[:, ['H', 'H_C', 'H_S']].plot()
    # in_data['S_dn'].plot(ax=ax, kind='area', legend=True)

    return

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        config_file = args[1]
    print('Run pyTSEB with configuration file = ' + str(config_file))
    run_TSEB_from_config_file(config_file)
