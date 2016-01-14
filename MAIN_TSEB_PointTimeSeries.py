#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script to run TSEB over a timeseries of point measurements

Created on Dec 29 2015
@author: Hector Nieto

Modified on Jan 13 2016
@author: Hector Nieto

"""

import time
from pyTSEB import PyTSEB
configFile='Config_PointTimeSeries.txt'

def RunTSEBFromConfigFile(configFile):
    # Open a log file in the working directory
    # Create a class instance from PyTSEB
    setup=PyTSEB()
    # Get the data from the widgets
    configData=setup.parseInputConfig(configFile,isImage=False)
    setup.GetDataTSEB(configData,isImage=False)
    setup.RunTSEBPointSeries()
    return

if __name__=='__main__':
    import sys
    args=sys.argv
    if len(args)>1:
        configFile=args[1]
    print('Run pyTSEB with configuration file = '+str(configFile))
    RunTSEBFromConfigFile(configFile)
