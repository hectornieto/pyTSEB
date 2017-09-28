# This file is part PyTSEB, consisting of of high level pyTSEB scripting
# Copyright 2016 Hector Nieto and contributors listed in the README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Created on Thu Jan  7 16:37:45 2016
@author: Hector Nieto (hnieto@ias.csic.es)

Modified on Jan  27 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the class object for configuring and running TSEB for both
an image with constant meteorology forcing and a time-series of tabulated data.

EXAMPLES
========
The easiest way to get a feeling of TSEB and its configuration is throuh the ipython/jupyter notebooks.

Jupyter notebook pyTSEB GUI
---------------------------
To configure TSEB for processing a time series of tabulated data, type in a ipython terminal or a jupyter notebook.

.. code-block:: ipython

    from TSEB_IPython_Interface import TSEB_IPython_Interface # Import IPython TSEB interface
    setup=TSEB_IPython_Interface() # Create the setup instance from the interface class object
    setup.PointTimeSeriesWidget() # Launches the GUI

then to run pyTSEB.

.. code-block:: ipython

    setup.GetDataTSEBWidgets(isImage = False) # Get the data from the widgets
    setup.RunTSEB(isImage = False) # Run TSEB

Similarly, to configure and run TSEB for an image.

.. code-block:: ipython

    from TSEB_IPython_Interface import TSEB_IPython_Interface # Import IPython TSEB interface
    setup=TSEB_IPython_Interface() # Create the setup instance from the interface class object
    setup.LocalImageWidget() # Launches the GUI
    setup.GetDataTSEBWidgets(isImage = True) # Get the data from the widgets
    setup.RunTSEB(isImage = True) # Run TSEB

Parsing directly a configuration file
-------------------------------------
You can also parse direcly into TSEB a configuration file previouly created.

>>> from TSEB_ConfigFile_Interface import TSEB_ConfigFile_Interface # Import Configuration File TSEB interface
>>> tseb=TSEB_ConfigFile_Interface()
>>> configData=tseb.parseInputConfig(configFile,isImage=True) # Read the data from the configuration file into a python dictionary
>>> tseb.GetDataTSEB(configData,isImage=True) # Parse the data from the dictionary to TSEB
>>> tseb.RunTSEB(isImage=True)

see the guidelines for input and configuration file preparation in :doc:`README_Notebooks`.

"""

from os.path import splitext, dirname, exists
from os import mkdir
import ast

import gdal
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from pyTSEB import TSEB
import pyTSEB.meteo_utils as met
import pyTSEB.net_radiation as rad
import pyTSEB.resistances as res
import pyTSEB.clumping_index as CI


class PyTSEB():

    def __init__(self, parameters):
        self.p = parameters
        
        # Model description parameters
        self.model_type = self.p['model']
        self.resistance_form = self.p['resistance_form']        
        self.res_params = {}
        self.G_form = self.p['G_form']

    def run_TSEB_local_image(self):
        ''' Runs TSEB for all the pixel in an image'''

        #======================================
        # Process the input

        # Create an input dictionary
        in_data = dict()

        if 'subset' in self.p:
            subset = ast.literal_eval(self.p['subset'])
        else:
            subset = []
        
        # Open the LST data according to the model
        try:
            fid = gdal.Open(self.p['T_R1'], gdal.GA_ReadOnly)
            prj = fid.GetProjection()
            geo = fid.GetGeoTransform()
            if subset:                   
                T_R1 = fid.GetRasterBand(1).ReadAsArray(subset[0],
                                                        subset[1],
                                                        subset[2],
                                                        subset[3])
                geo = [geo[0]+subset[0]*geo[1], geo[1], geo[2], 
                       geo[3]+subset[1]*geo[5], geo[4], geo[5]]
            else:
                T_R1 = fid.GetRasterBand(1).ReadAsArray()
            dims = np.shape(T_R1)

            # In case of TSEB_PT or DTD models save noon temperature
            if self.model_type == 'TSEB_PT' or self.model_type == 'DTD':
                in_data['T_R1'] = T_R1
            
            # In case of TSEB_2T save both component temperatures
            if self.model_type == 'TSEB_2T':
                in_data['T_C'] =  T_R1
                if subset:
                    in_data['T_S'] = fid.GetRasterBand(2).ReadAsArray(subset[0],
                                                                      subset[1],
                                                                      subset[2],
                                                                      subset[3])
                else:
                   in_data['T_S'] = fid.GetRasterBand(2).ReadAsArray() 
                
            fid = None
        except:
            print('Error reading sunrise LST file ' + str(self.p['T_R0']))
            fid = None
            return
                  
        # In case of DTD also need to read the sunrise/night LST
        if self.model_type == 'DTD': 
            success, in_data['T_R0'] = self._open_GDAL_image(
                self.p['T_R0'], dims, 'Sunrise LST', subset)
            if not success:
                return
            
        # Read the image mosaic and get the LAI
        success, in_data['LAI'] = self._open_GDAL_image(
            self.p['LAI'], dims, 'Leaf Area Index', subset)
        if not success:
            return
        # Read the image View Zenith Angle
        success, in_data['VZA'] = self._open_GDAL_image(
            self.p['VZA'], dims, 'View Zenith Angle', subset)
        if not success:
            return
        # Read the fractional cover data
        success, in_data['f_c'] = self._open_GDAL_image(
            self.p['f_c'], dims, 'Fractional Cover', subset)
        if not success:
            return
        # Read the Canopy Height data
        success, in_data['h_C'] = self._open_GDAL_image(
            self.p['h_C'], dims, 'Canopy Height', subset)
        if not success:
            return
        # Read the canopy witdth ratio
        success, in_data['w_C'] = self._open_GDAL_image(
            self.p['w_C'], dims, 'Canopy Width Ratio', subset)
        if not success:
            return
        # Read the Green fraction
        success, in_data['f_g'] = self._open_GDAL_image(
            self.p['f_g'], dims, 'Green Fraction', subset)
        if not success:
            return
        # Read landcover
        success, in_data['landcover'] = self._open_GDAL_image(
            self.p['landcover'], dims, 'Landcover', subset)
        if not success:
            return
        # Read leaf angle distribution
        success, in_data['x_LAD'] = self._open_GDAL_image(
            self.p['x_LAD'], dims, 'Leaf Angle Distribution', subset)
        if not success:
            return
        # Read initial alpha_PT
        success, in_data['alpha_PT'] = self._open_GDAL_image(
            self.p['alpha_PT'], dims, 'Initial alpha_PT', subset)
        if not success:
            return
        
        # Read spectral properties
        success, in_data['rho_vis_C'] = self._open_GDAL_image(
            self.p['rho_vis_C'], dims, 'Leaf PAR Reflectance', subset)
        if not success:
            return
        success, in_data['tau_vis_C'] = self._open_GDAL_image(
            self.p['tau_vis_C'], dims, 'Leaf PAR Transmitance', subset)
        if not success:
            return
        success, in_data['rho_nir_C'] = self._open_GDAL_image(
            self.p['rho_nir_C'], dims, 'Leaf NIR Reflectance', subset)
        if not success:
            return
        success, in_data['tau_nir_C'] = self._open_GDAL_image(
            self.p['tau_nir_C'], dims, 'Leaf NIR Transmitance', subset)
        if not success:
            return
        success, in_data['rho_vis_S'] = self._open_GDAL_image(
            self.p['rho_vis_S'], dims, 'Soil PAR Reflectance', subset)
        if not success:
            return
        success, in_data['rho_nir_S'] = self._open_GDAL_image(
            self.p['rho_nir_S'], dims, 'Soil NIR Reflectance', subset)
        if not success:
            return
        success, in_data['emis_C'] = self._open_GDAL_image(
            self.p['emis_C'], dims, 'Leaf Emissivity', subset)
        if not success:
            return
        success, in_data['emis_S'] = self._open_GDAL_image(
            self.p['emis_S'], dims, 'Soil Emissivity', subset)
        if not success:
            return
        
        # Calculate illumination conditions
        success, lat = self._open_GDAL_image(self.p['lat'], dims, 'Latitude', subset)
        if not success:
            return
        success, lon = self._open_GDAL_image(self.p['lon'], dims, 'Longitude', subset)
        if not success:
            return
        success, stdlon = self._open_GDAL_image(self.p['stdlon'], dims, 'Standard Longitude', subset)
        if not success:
            return
        success, in_data['time'] = self._open_GDAL_image(self.p['time'], dims, 'Time', subset)
        if not success:
            return
        success, doy = self._open_GDAL_image(self.p['DOY'], dims, 'DOY', subset)
        if not success:
            return
        in_data['SZA'], in_data['SAA'] = met.calc_sun_angles(
            lat, lon, stdlon, doy, in_data['time'])
        # Wind speed
        success, in_data['u'] = self._open_GDAL_image(
            self.p['u'], dims, 'Wind speed', subset)
        if not success:
            return
        # Vapour pressure
        success, in_data['ea'] = self._open_GDAL_image(
            self.p['ea'], dims, 'Vapour pressure', subset)
        if not success:
            return
        # Air pressure
        success, in_data['p'] = self._open_GDAL_image(
            self.p['p'], dims, 'Pressure', subset)
        # If pressure was not provided then estimate it based of altitude
        if not success:
            success, alt = self._open_GDAL_image(self.p['alt'], dims, 'Altitude', subset)
            if success:
                in_data['p'] = met.calc_pressure(alt)
            else:
                return
        success, in_data['S_dn'] = self._open_GDAL_image(
            self.p['S_dn'], dims, 'Shortwave irradiance', subset)
        if not success:
            return
        # Wind speed measurement height
        success, in_data['z_u'] = self._open_GDAL_image(
            self.p['z_u'], dims, 'Wind speed height', subset)
        if not success:
            return
        # Air temperature mesurement height
        success, in_data['z_T'] = self._open_GDAL_image(
            self.p['z_T'], dims, 'Air temperature height', subset)
        if not success:
            return
        # Leaf width
        success, in_data['leaf_width'] = self._open_GDAL_image(
            self.p['leaf_width'], dims, 'Leaf Width', subset)
        if not success:
            return
        # Soil roughness
        success, in_data['z0_soil'] = self._open_GDAL_image(
            self.p['z0_soil'], dims, 'Soil Roughness', subset)
        if not success:
            return 

        # Incoming long wave radiation
        success, in_data['T_A1'] = self._open_GDAL_image(
            self.p['T_A1'], dims, 'Air Temperature', subset)
        if not success:
            return
        if self.model_type == 'DTD':
            success, in_data['T_A0'] = self._open_GDAL_image(
                self.p['T_A0'], dims, 'Air Temperature at time 0', subset)
        if not success:
            return
        success, in_data['L_dn'] = self._open_GDAL_image(
            self.p['L_dn'], dims, 'Longwave irradiance', subset)
        # If longwave irradiance was not provided then estimate it based on air
        # temperature and humidity
        if not success:
            emisAtm = rad.calc_emiss_atm(in_data['ea'], in_data['T_A1'])
            in_data['L_dn'] = emisAtm * met.calc_stephan_boltzmann(in_data['T_A1'])
            
        # Open the processing maks and get the id for the cells to process
        if self.p['input_mask'] != '0':
            success, mask = self._open_GDAL_image(
                self.p['input_mask'], dims, 'input mask', subset) 
            if not success:
                print ("Please set input_mask=0 for processing the whole image.")
                return
        # Otherwise create mask from landcover array
        else:
            mask = np.ones(dims)
            mask[np.logical_or.reduce((in_data['landcover'] == res.WATER, 
                                       in_data['landcover'] == res.URBAN,
                                       in_data['landcover'] == res.SNOW))] = 0


        # Get the Soil Heat flux if G_form includes the option of measured G
        # Constant G or constant ratio of soil reaching radiation
        if self.G_form[0][0] == 0 or self.G_form[0][0] == 1:  
            success, self.G_form[1] = self._open_GDAL_image(
                                            self.G_form[1], dims, 'G', subset)
            if not success:
                return   
        # Santanello and Friedls G
        elif self.G_form[0][0] == 2:  
            # Set the time in the G_form flag to compute the Santanello and
            # Friedl G
            self.G_form[1] = in_data['time']
        
        # Set the Kustas and Norman resistance parameters          
        if self.resistance_form == 0:
            success, self.res_params['KN_b'] = self._open_GDAL_image(
                self.p['KN_b'], dims, 'Resistance parameter b', subset)
            if not success:
                return
            success, self.res_params['KN_c'] = self._open_GDAL_image(
                self.p['KN_c'], dims, 'Resistance parameter c', subset)
            if not success:
                return 
            success, self.res_params['KN_C_dash'] = self._open_GDAL_image(
                self.p['KN_C_dash'], dims, 'Resistance parameter C\'', subset)
            if not success:
                return         
        
        #======================================
        # Run the chosen model
        
        out_data = self.run_TSEB(in_data, mask)


        #======================================
        # Save output files

        # Output variables saved in images
        self.fields = ('H1', 'LE1', 'R_n1', 'G1')
        # Ancillary output variables
        self.anc_fields = (
            'H_C1',
            'LE_C1',
            'LE_partition',
            'T_C1',
            'T_S1',
            'R_ns1',
            'R_nl1',
            'delta_R_n1',
            'u_friction',
            'L',
            'R_S1',
            'R_x1',
            'R_A1',
            'flag')
        outdir = dirname(self.p['output_file'])
        if not exists(outdir):
            mkdir(outdir)
        self._write_raster_output(
            self.p['output_file'],
            out_data,
            geo,
            prj,
            self.fields)
        outputfile = splitext(self.p['output_file'])[0] + '_ancillary' + \
                     splitext(self.p['output_file'])[1]
        self._write_raster_output(
            outputfile,
            out_data,
            geo,
            prj,
            self.anc_fields)
        print('Saved Files')
        
        return in_data, out_data


    def run_TSEB_point_series_array(self):
        ''' Runs TSEB for all the dates in point time-series'''

        def compose_date(
                years,
                months=1,
                days=1,
                weeks=None,
                hours=None,
                minutes=None,
                seconds=None,
                milliseconds=None,
                microseconds=None,
                nanoseconds=None):
            ''' Taken from http://stackoverflow.com/questions/34258892/converting-year-and-day-of-year-into-datetime-index-in-pandas'''
            years = np.asarray(years) - 1970
            months = np.asarray(months) - 1
            days = np.asarray(days) - 1
            types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
                     '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
            vals = (years, months, days, weeks, hours, minutes, seconds,
                    milliseconds, microseconds, nanoseconds)
            return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
                       if v is not None)

        #======================================
        # Process the input
        
        # Read input data from CSV file
        in_data = pd.read_csv(self.p['input_file'], delim_whitespace=True)
        in_data.index = compose_date(
            years=in_data['year'],
            days=in_data['DOY'],
            hours=in_data['time'],
            minutes=in_data['time'] % 1 * 60)

        # Check if all the required columns are present
        if not self._required_data_present(in_data):
            return None, None

        # Fill in data fields which might not be in the input file
        if 'SZA' not in in_data.columns:
            sza, _ = met.calc_sun_angles(self.p['lat'], self.p['lon'], self.p[
                                            'stdlon'], in_data['DOY'], in_data['time'])
            in_data['SZA'] = sza
        if 'SAA' not in in_data.columns:
            _, saa = met.calc_sun_angles(self.p['lat'], self.p['lon'], self.p[
                                            'stdlon'], in_data['DOY'], in_data['time'])
            in_data['SAA'] = saa
        if 'p' not in in_data.columns:  # Estimate barometric pressure from the altitude if not included in the table
            in_data['p'] = met.calc_pressure(self.p['alt'])
        if 'f_c' not in in_data.columns:  # Fractional cover
            in_data['f_c'] = self.p['f_c']  # Use default value
        if 'w_C' not in in_data.columns:  # Canopy width to height ratio
            in_data['w_C'] = self.p['w_C']  # Use default value
        if 'f_g' not in in_data.columns:  # Green fraction
            in_data['f_g'] = self.p['f_g']  # Use default value
        if 'rho_vis_C' not in in_data.columns:
            in_data['rho_vis_C'] = self.p['rho_vis_C']
        if 'tau_vis_C' not in in_data.columns:
            in_data['tau_vis_C'] = self.p['tau_vis_C']
        if 'rho_nir_C' not in in_data.columns:
            in_data['rho_nir_C'] = self.p['rho_nir_C']
        if 'tau_nir_C' not in in_data.columns:
            in_data['tau_nir_C'] = self.p['tau_nir_C']
        if 'rho_vis_S' not in in_data.columns:
            in_data['rho_vis_S'] = self.p['rho_vis_S']
        if 'rho_nir_S' not in in_data.columns:
            in_data['rho_nir_S'] = self.p['rho_nir_S']
        if 'emis_C' not in in_data.columns:
            in_data['emis_C'] = self.p['emis_C']
        if 'emis_S' not in in_data.columns:
            in_data['emis_S'] = self.p['emis_S'] 
            
        # Fill in other data fields from the parameter file
        in_data['landcover'] = self.p['landcover']
        in_data['z_u'] = self.p['z_u']
        in_data['z_T'] = self.p['z_T']
        in_data['leaf_width'] = self.p['leaf_width']
        in_data['z0_soil'] = self.p['z0_soil']
        in_data['alpha_PT'] = self.p['alpha_PT']
        in_data['x_LAD'] = self.p['x_LAD']
        

        # Incoming long wave radiation
        if 'L_dn' not in in_data.columns:
            # If longwave irradiance was not provided then estimate it based on air
            # temperature and humidity
            emisAtm = rad.calc_emiss_atm(in_data['ea'], in_data['T_A1'])
            in_data['L_dn'] = emisAtm * met.calc_stephan_boltzmann(in_data['T_A1'])

        # Get the Soil Heat flux if G_form includes the option of measured G
        dims = in_data['LAI'].shape         
        if self.G_form[0][0] == 0:  # Constant G
            if 'G' in in_data.columns:
                self.G_form[1] = in_data['G']
        elif self.G_form[0][0] == 1:
            self.G_form[1] = np.ones(dims) * self.G_form[1]
        elif self.G_form[0][0] == 2:  # Santanello and Friedls G
            # Set the time in the G_form flag to compute the Santanello and
            # Friedl G
            self.G_form[1] = in_data['time']

        # Set the Kustas and Norman resistance parameters     
        if self.resistance_form == 0:
            self.res_params['KN_b'] =  np.ones(dims) * self.p['KN_b']
            self.res_params['KN_c'] = np.ones(dims) * self.p['KN_c']
            self.res_params['KN_C_dash'] = np.ones(dims) * self.p['KN_C_dash']

        #======================================
        # Run the chosen model

        out_data = self.run_TSEB(in_data)
        out_data = pd.DataFrame(data = np.stack(out_data.values()).T,
                                index = in_data.index,
                                columns = out_data.keys())

        #======================================
        # Save output file

        # Output Headers
        outputTxtFieldNames = [
            'Year',
            'DOY',
            'Time',
            'LAI',
            'f_g',
            'VZA',
            'SZA',
            'SAA',
            'L_dn',
            'Rn_model',
            'Rn_sw_veg',
            'Rn_sw_soil',
            'Rn_lw_veg',
            'Rn_lw_soil',
            'T_C',
            'T_S',
            'T_AC',
            'LE_model',
            'H_model',
            'LE_C',
            'H_C',
            'LE_S',
            'H_S',
            'G_model',
            'R_S',
            'R_x',
            'R_A',
            'u_friction',
            'L',
            'Skyl',
            'z_0M',
            'd_0',
            'flag']

        # Create the ouput directory if it doesn't exist
        outdir = dirname(self.p['output_file'])
        if not exists(outdir):
            mkdir(outdir)

        # Write the data
        csvData = pd.concat([in_data[['year',
                                     'DOY',
                                     'time',
                                     'LAI',
                                     'f_g',
                                     'VZA',
                                     'SZA',
                                     'SAA',
                                     'L_dn']],
                             out_data[['R_n1',
                                      'Sn_C1',
                                      'Sn_S1',
                                      'Ln_C1',
                                      'Ln_S1',
                                      'T_C1',
                                      'T_S1',
                                      'T_AC1',
                                      'LE1',
                                      'H1',
                                      'LE_C1',
                                      'H_C1',
                                      'LE_S1',
                                      'H_S1',
                                      'G1',
                                      'R_S1',
                                      'R_x1',
                                      'R_A1',
                                      'u_friction',
                                      'L',
                                      'Skyl',
                                      'z_0M',
                                      'd_0',
                                      'flag']]],
                            axis=1)
        csvData.to_csv(
            self.p['output_file'],
            sep='\t',
            index=False,
            header=outputTxtFieldNames)

        print('Done')

        return in_data, out_data


    def run_TSEB(self, in_data, mask = None):
        
        print("Processing...")
        
        if mask is None:
            mask = np.ones(in_data['LAI'].shape)
        
        # Create the output dictionary
        out_data = dict()
        for field in self._get_output_structure():
            out_data[field] = np.zeros(in_data['LAI'].shape) + np.NaN
        
        # Esimate diffuse and direct irradiance
        difvis, difnir, fvis, fnir = rad.calc_difuse_ratio(
            in_data['S_dn'], in_data['SZA'], press=in_data['p'])
        out_data['fvis'] = fvis
        out_data['fnir'] = fnir
        out_data['Skyl'] = difvis * fvis + difnir * fnir
        out_data['S_dn_dir'] = in_data['S_dn'] * (1.0 - out_data['Skyl'])
        out_data['S_dn_dif'] = in_data['S_dn'] * out_data['Skyl']        
        
        #======================================
        # First process bare soil cases
        
        noVegPixels = in_data['LAI'] <= 0
        noVegPixels = np.logical_or.reduce(
            (in_data['f_c'] <= 0.01, 
             in_data['LAI'] <= 0, 
             np.isnan(in_data['LAI'])))
        #in_data['LAI'][noVegPixels] = 0
        #in_data['f_c'][noVegPixels] = 0
        i = np.array(np.logical_and(noVegPixels, mask == 1))
        
        # Calculate roughness
        out_data['z_0M'][i] = in_data['z0_soil'][i]
        out_data['d_0'][i] = 5 * out_data['z_0M'][i]

        # Net shortwave radition for bare soil
        spectraGrdOSEB = out_data['fvis']  * \
            in_data['rho_vis_S'] + out_data['fnir'] * in_data['rho_nir_S']
        out_data['Sn_S1'][i] = (1. - spectraGrdOSEB[i]) * \
            (out_data['S_dn_dir'][i] + out_data['S_dn_dif'][i])

        # Other fluxes for bare soil
        if self.model_type == 'DTD':
            T_S_K = in_data['T_R1'][i]
            T0_K = (in_data['T_R0'][i], in_data['T_A0'][i])
        elif self.model_type == 'TSEB_PT':
            T_S_K = in_data['T_R1'][i]
            T0_K = []
        else:
            T_S_K = in_data['T_S'][i]
            T0_K = []
        [out_data['flag'][i],
         out_data['Ln_S1'][i],
         out_data['LE_S1'][i],
         out_data['H_S1'][i],
         out_data['G1'][i],
         out_data['R_A1'][i],
         out_data['u_friction'][i],
         out_data['L'][i],
         out_data['n_iterations'][i]] = TSEB.OSEB(T_S_K,
                                                  in_data['T_A1'][i],
                                                  in_data['u'][i],
                                                  in_data['ea'][i],
                                                  in_data['p'][i],
                                                  out_data['Sn_S1'][i],
                                                  in_data['L_dn'][i],
                                                  in_data['emis_S'][i],
                                                  out_data['z_0M'][i],
                                                  out_data['d_0'][i],
                                                  in_data['z_u'][i],
                                                  in_data['z_T'][i],
                                                  calcG_params=[self.G_form[0],
                                                               self.G_form[1][i]],
                                                  T0_K=T0_K)
        
        # Set canopy fluxes to 0
        out_data['Sn_C1'][i] = 0.0                                          
        out_data['Ln_C1'][i] = 0.0
        out_data['LE_C1'][i] = 0.0
        out_data['H_C1'][i] = 0.0
        
        #======================================
        # Then process vegetated cases
        
        i = np.array(np.logical_and(~noVegPixels, mask == 1))

        # Calculate roughness
        out_data['z_0M'][i], out_data['d_0'][i] = \
            res.calc_roughness(in_data['LAI'][i], 
                               in_data['h_C'][i], 
                               w_C = in_data['w_C'][i], 
                               landcover = in_data['landcover'][i],
                               f_c = in_data['f_c'][i])

        # Net shortwave radiation for vegetation
        F = np.zeros(in_data['LAI'].shape)
        F[i] = in_data['LAI'][i] / in_data['f_c'][i]
        # Clumping index
        omega0, Omega = np.zeros(in_data['LAI'].shape), np.zeros(in_data['LAI'].shape)
        omega0[i] = CI.calc_omega0_Kustas(
                        in_data['LAI'][i],
                        in_data['f_c'][i],
                        x_LAD=in_data['x_LAD'][i],
                        isLAIeff=True)
        if self.p['calc_row'][0] == 0:  # randomly placed canopies
            Omega[i] = CI.calc_omega_Kustas(
                omega0[i], in_data['SZA'][i], w_C=in_data['w_C'][i])
        else:
            Omega[i] = CI.calc_omega_Kustas(
                omega0[i], in_data['SZA'][i], w_C=in_data['w_C'][i])
        LAI_eff = F * Omega
        [out_data['Sn_C1'][i],
         out_data['Sn_S1'][i]] = rad.calc_Sn_Campbell(in_data['LAI'][i],
                                                     in_data['SZA'][i],
                                                     out_data['S_dn_dir'][i],
                                                     out_data['S_dn_dif'][i],
                                                     out_data['fvis'][i],
                                                     out_data['fnir'][i],
                                                     in_data['rho_vis_C'][i],
                                                     in_data['tau_vis_C'][i],
                                                     in_data['rho_nir_C'][i],
                                                     in_data['tau_nir_C'][i],
                                                     in_data['rho_vis_S'][i],
                                                     in_data['rho_nir_S'][i],
                                                     x_LAD=in_data['x_LAD'][i],
                                                     LAI_eff=LAI_eff[i])

        # Model settings
        calcG_params = [self.G_form[0], self.G_form[1][i]]
        resistance_form = [self.resistance_form, 
                           {k: self.res_params[k][i] for k in self.res_params}]

        # Other fluxes for vegetation
        if self.model_type == 'DTD':
            [out_data['flag'][i], out_data['T_S1'][i], out_data['T_C1'][i],
             out_data['T_AC1'][i], out_data['Ln_S1'][i], out_data['Ln_C1'][i],
             out_data['LE_C1'][i], out_data['H_C1'][i], out_data['LE_S1'][i],
             out_data['H_S1'][i], out_data['G1'][i], out_data['R_S1'][i],
             out_data['R_x1'][i], out_data['R_A1'][i], out_data['u_friction'][i],
             out_data['L'][i], out_data['Ri'], out_data['n_iterations'][i]] = \
                     TSEB.DTD(in_data['T_R0'][i],
                              in_data['T_R1'][i],
                              in_data['VZA'][i],
                              in_data['T_A0'][i],
                              in_data['T_A1'][i],
                              in_data['u'][i],
                              in_data['ea'][i],
                              in_data['p'][i],
                              out_data['Sn_C1'][i],
                              out_data['Sn_S1'][i],
                              in_data['L_dn'][i],
                              in_data['LAI'][i],
                              in_data['h_C'][i],
                              in_data['emis_C'][i],
                              in_data['emis_S'][i],
                              out_data['z_0M'][i],
                              out_data['d_0'][i],
                              in_data['z_u'][i],
                              in_data['z_T'][i],
                              f_c=in_data['f_c'][i],
                              w_C=in_data['w_C'][i],
                              f_g=in_data['f_g'][i],
                              leaf_width=in_data['leaf_width'][i],
                              z0_soil=in_data['z0_soil'][i],
                              alpha_PT=in_data['alpha_PT'][i],
                              x_LAD=in_data['x_LAD'][i],
                              calcG_params=calcG_params,
                              resistance_form=resistance_form)

        elif self.model_type == 'TSEB_PT':
            [out_data['flag'][i], out_data['T_S1'][i], out_data['T_C1'][i],
             out_data['T_AC1'][i], out_data['Ln_S1'][i], out_data['Ln_C1'][i],
             out_data['LE_C1'][i], out_data['H_C1'][i], out_data['LE_S1'][i],
             out_data['H_S1'][i], out_data['G1'][i], out_data['R_S1'][i],
             out_data['R_x1'][i], out_data['R_A1'][i], out_data['u_friction'][i],
             out_data['L'][i], out_data['n_iterations'][i]] = \
                     TSEB.TSEB_PT(in_data['T_R1'][i],
                                  in_data['VZA'][i],
                                  in_data['T_A1'][i],
                                  in_data['u'][i],
                                  in_data['ea'][i],
                                  in_data['p'][i],
                                  out_data['Sn_C1'][i],
                                  out_data['Sn_S1'][i],
                                  in_data['L_dn'][i],
                                  in_data['LAI'][i],
                                  in_data['h_C'][i],
                                  in_data['emis_C'][i],
                                  in_data['emis_S'][i],
                                  out_data['z_0M'][i],
                                  out_data['d_0'][i],
                                  in_data['z_u'][i],
                                  in_data['z_T'][i],
                                  f_c=in_data['f_c'][i],
                                  f_g=in_data['f_g'][i],
                                  w_C=in_data['w_C'][i],
                                  leaf_width=in_data['leaf_width'][i],
                                  z0_soil=in_data['z0_soil'][i],
                                  alpha_PT=in_data['alpha_PT'][i],
                                  x_LAD=in_data['x_LAD'][i],
                                  calcG_params=calcG_params,
                                  resistance_form=resistance_form)

        elif self.model_type == 'TSEB_2T':
            # Run TSEB with the component temperatures T_S and T_C
            [out_data['flag'][i], out_data['T_AC1'][i], out_data['Ln_S1'][i], 
             out_data['Ln_C1'][i], out_data['LE_C1'][i], out_data['H_C1'][i], 
             out_data['LE_S1'][i], out_data['H_S1'][i], out_data['G1'][i], 
             out_data['R_S1'][i], out_data['R_x1'][i], out_data['R_A1'][i], 
             out_data['u_friction'][i], out_data['L'][i], out_data['n_iterations'][i]] = \
                     TSEB.TSEB_2T(in_data['T_C'][i],
                                  in_data['T_S'][i],
                                  in_data['T_A1'][i],
                                  in_data['u'][i],
                                  in_data['ea'][i],
                                  in_data['p'][i],
                                  out_data['Sn_C1'][i],
                                  out_data['Sn_S1'][i],
                                  in_data['L_dn'][i],
                                  in_data['LAI'][i],
                                  in_data['h_C'][i],
                                  in_data['emis_C'][i],
                                  in_data['emis_S'][i],
                                  out_data['z_0M'][i],
                                  out_data['d_0'][i],
                                  in_data['z_u'][i],
                                  in_data['z_T'][i],
                                  f_c=in_data['f_c'][i],
                                  f_g=in_data['f_g'][i],
                                  w_C=in_data['w_C'][i],
                                  leaf_width=in_data['leaf_width'][i],
                                  z0_soil=in_data['z0_soil'][i],
                                  alpha_PT=in_data['alpha_PT'][i],
                                  x_LAD=in_data['x_LAD'][i],
                                  calcG_params=calcG_params,
                                  resistance_form=resistance_form)

        # Calculate the bulk fluxes
        out_data['LE1'] = out_data['LE_C1'] + out_data['LE_S1']
        out_data['LE_partition'] = out_data['LE_C1'] / out_data['LE1'] 
        out_data['H1'] = out_data['H_C1'] + out_data['H_S1']
        out_data['R_ns1'] = out_data['Sn_C1'] + out_data['Sn_S1']
        out_data['R_nl1'] = out_data['Ln_C1'] + out_data['Ln_S1']
        out_data['R_n1'] = out_data['R_ns1'] + out_data['R_nl1']
        out_data['delta_R_n1'] = out_data['Sn_C1'] + out_data['Ln_C1']
        
        print("Finished processing!")        
        return out_data

    def _open_GDAL_image(self, inputString, dims, variable, subset = []):
        '''Open a GDAL image and returns and array with its first band'''

        if inputString == "":
            return False, None
      
        success = True
        array = None
        try:
            array = np.zeros(dims) + float(inputString)
        except:
            try:
                fid = gdal.Open(inputString, gdal.GA_ReadOnly)
                if subset:
                    array = fid.GetRasterBand(1).ReadAsArray(subset[0],
                                                             subset[1],
                                                             subset[2],
                                                             subset[3])
                else:
                    array = fid.GetRasterBand(1).ReadAsArray()
            except:
                print(
                    'ERROR: file read ' +
                    str(inputString) +
                    '\n Please type a valid file name or a numeric value for ' +
                    variable)
                success = False
            finally:
                fid = None
                
        return success, array

    def _write_raster_output(self, outfile, output, geo, prj, fields):
        '''Writes the arrays of an output dictionary which keys match the list 
           in fields to a raster file '''

        # If the output file has .nc extension then save it as netCDF,
        # otherwise assume that the output should be a GeoTIFF
        ext = splitext(outfile)[1]
        if ext.lower() == ".nc":
            driver = "netCDF"
            opt = ["FORMAT=NC2"]
            is_netCDF = True
        else:
            driver = "GTiff"
            opt = []
            is_netCDF = False

        # Save the data using GDAL
        rows, cols = np.shape(output['H1'])
        driver = gdal.GetDriverByName(driver)
        nbands = len(fields)
        ds = driver.Create(outfile, cols, rows, nbands, gdal.GDT_Float32, opt)
        ds.SetGeoTransform(geo)
        ds.SetProjection(prj)
        for i, field in enumerate(fields):
            band = ds.GetRasterBand(i + 1)
            band.SetNoDataValue(np.NaN)
            band.WriteArray(output[field])
            band.FlushCache()
        ds.FlushCache()
        ds = None
        
        # In case of netCDF format use netCDF4 module to assign proper names 
        # to variables (GDAL can't do this). Also it seems that GDAL has
        # problems assigning projection to all the bands so fix that.
        if is_netCDF:
            ds = Dataset(outfile, 'a')
            grid_mapping = ds["Band1"].grid_mapping
            for i, field in enumerate(fields):
                ds.renameVariable("Band"+str(i+1), field)
                ds[field].grid_mapping = grid_mapping
            ds.close()
            
    def _get_output_structure(self):
        ''' Output fields in TSEB'''
        
        outputStructure = (
            # resistances
            'R_A1', # resistance to heat transport in the surface layer (s/m) at time t1
            'R_x1', # resistance to heat transport in the canopy surface layer (s/m) at time t1
            'R_S1', # resistance to heat transport from the soil surface (s/m) at time t1 fluxes
            # Energy fluxes            
            'R_n1',  # net radiation reaching the surface at time t1
            'R_ns1',  # net shortwave radiation reaching the surface at time t1
            'R_nl1',  # net longwave radiation reaching the surface at time t1
            'delta_R_n1',  # net radiation divergence in the canopy at time t1
            'Sn_S1', # Shortwave radiation reaching the soil at time t1
            'Sn_C1', # Shortwave radiation intercepted by the canopy at time t1
            'Ln_S1', # Longwave radiation reaching the soil at time t1
            'Ln_C1', # Longwave radiation intercepted by the canopy at time t1
            'H_C1',  # canopy sensible heat flux (W/m^2) at time t1
            'H_S1',  # soil sensible heat flux (W/m^2) at time t1
            'H1',  # total sensible heat flux (W/m^2) at time t1
            'G1',  # ground heat flux (W/m^2) at time t1
            'LE_C1',  # canopy latent heat flux (W/m^2) at time t1
            'LE_S1',  # soil latent heat flux (W/m^2) at time t1
            'LE1',  # total latent heat flux (W/m^2) at time t1
            'LE_partition',  # Latent Heat Flux Partition (LEc/LE) at time t1
            # temperatures (might not be accurate)
            'T_C1',  # canopy temperature at time t1 (deg C)
            'T_S1',  # soil temperature at time t1 (deg C)
            'T_AC1',  # air temperature at the canopy interface at time t1 (deg C)
            # miscaleneous
            'albedo1',    # surface albedo (Rs_out/Rs_in)
            'omega0',  # nadir view vegetation clumping factor
            'alpha',  # the priestly Taylor factor
            'Ri',  # Richardson number at time t1
            'L',  # Monin Obukhov Length at time t1
            'u_friction',  # Friction velocity
            'theta_s1',  # Sun zenith angle at time t1
            'F', # Leaf Area Index    
            'z_0M', # Aerodynamic roughness length for momentum trasport (m)
            'd_0', # Zero-plane displacement height (m)
            'Skyl', 
            'flag', # Quality flag
            'n_iterations') # Number of iterations before model converged to stable value 
            

        return outputStructure

    def _required_data_present(self, in_data):
        '''Checks that all the data required for TSEB is contained in an input ascci table'''

        # Mandatory Input Fields
        MandatoryFields_TSEB_PT = (
            'year',
            'DOY',
            'time',
            'T_R1',
            'VZA',
            'T_A1',
            'u',
            'ea',
            'S_dn',
            'LAI',
            'h_C')
        MandatoryFields_DTD = (
            'year',
            'DOY',
            'time',
            'T_R0',
            'T_R1',
            'VZA',
            'T_A0',
            'T_A1',
            'u',
            'ea',
            'S_dn',
            'LAI',
            'h_C')
        MandatoryFields_TSEB_2T = (
            'year',
            'DOY',
            'time',
            'T_C',
            'T_S',
            'T_A1',
            'u',
            'ea',
            'S_dn',
            'LAI',
            'h_C')

        # Check that all mandatory input variables exist
        if self.model_type == 'TSEB_PT':
            missing = set(MandatoryFields_TSEB_PT) - (set(in_data.columns))
        elif self.model_type == 'DTD':
            missing = set(MandatoryFields_DTD) - (set(in_data.columns))
        elif self.model_type == 'TSEB_2T':
            missing = set(MandatoryFields_TSEB_2T) - (set(in_data.columns))
        else:
            print('Not valid TSEB model, check your configuration file')
            return False

        if missing:
            print ('ERROR: ' +
                   str(list(missing)) +
                   ' not found in file ' +
                   self.p['input_file'])
            return False
        else:
            return True

