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
@author: Hector Nieto (hector.nieto@ica.csic.es)

DESCRIPTION
===========
This package contains the class object for configuring and running TSEB for both
an image with constant meteorology forcing and a time-series of tabulated data.

EXAMPLES
========
The easiest way to get a feeling of TSEB and its configuration is throuh the ipython/jupyter
notebooks.

Jupyter notebook pyTSEB GUI
---------------------------
To configure TSEB for processing a time series of tabulated data, type in a ipython terminal or a
jupyter notebook.

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

from os.path import join, splitext, dirname, basename, exists
from os import mkdir
from collections import OrderedDict
import math

from osgeo import gdal, ogr, osr
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from . import TSEB
from . import meteo_utils as met
from . import net_radiation as rad
from . import resistances as res
from . import clumping_index as CI
from . import energy_combination_ET as pet
from . import dis_TSEB


# Constants for indicating whether model output field should be saved to file
S_N = 0  # Save Not
S_P = 1  # Save as Primary output
S_A = 2  # Save as Ancillary output


class PyTSEB(object):

    def __init__(self, parameters):
        self.p = parameters

        # Model description parameters
        self.model_type = self.p['model']
        self.resistance_form = self.p['resistance_form']
        self.res_params = {}
        self.G_form = self.p['G_form']
        self.water_stress = self.p['water_stress']
        self.calc_daily_ET = False

    def process_local_image(self):
        ''' Prepare input data and calculate energy fluxes for all the pixel in an image.

        Parameters
        ----------
        None

        Returns
        -------
        in_data : dict
            All the input data coming into the model.
        out_data : dict
            All the output data coming out of the model.
        '''

        # ======================================
        # Process the input

        # Create an input dictionary
        in_data = dict()
        temp_data = dict()
        res_params = dict()
        input_fields = self._get_input_structure()

        # Get projection, geo transform of the input dataset, or its subset if specified.
        # It is assumed that all the input rasters have exactly the same projection, dimensions and
        # resolution.
        try:
            field = list(input_fields)[0]
            fid = gdal.Open(self.p[field], gdal.GA_ReadOnly)
            self.prj = fid.GetProjection()
            self.geo = fid.GetGeoTransform()
            dims = (fid.RasterYSize, fid.RasterXSize)
            fid = None
            self.subset = []
            if "subset" in self.p:
                self.subset, self.geo = self._get_subset(self.p["subset"], self.prj, self.geo)
                if self.subset[3] <= 0 or self.subset[2] <= 0 or\
                   self.subset[0] >= dims[1] or self.subset[1] >= dims[0]:
                    print("ERROR: Requested subset does not intersect the data extent.")
                    return
                if self.subset[1] + self.subset[3] > dims[0] or\
                   self.subset[0] + self.subset[2] > dims[1]:
                    print("WARNING: Requested subset extends beyond the data extent.")
                    self.subset[3] = min(self.subset[3],  dims[0] - self.subset[1])
                    self.subset[2] = min(self.subset[2],  dims[1] - self.subset[0])
                dims = (self.subset[3], self.subset[2])
        except KeyError:
            print('Error reading ' + input_fields[field])
            fid = None
            return

        # Process all input fields
        for field in list(input_fields):
            # Some fields might need special treatment
            if field in ["lat", "lon", "stdlon", "DOY", "time"]:
                success, temp_data[field] = self._set_param_array(field, dims)
            elif field == "input_mask":
                if self.p['input_mask'] == '0':
                    # Create mask from landcover array
                    mask = np.ones(dims, np.int32)
                    mask[np.logical_or.reduce((in_data['landcover'] == res.WATER,
                                               in_data['landcover'] == res.URBAN,
                                               in_data['landcover'] == res.SNOW))] = 0
                    success = True
                else:
                    success, mask = self._set_param_array(field, dims)
            elif field in ['KN_b', 'KN_c', 'KN_c_dash']:
                success, res_params[field] = self._set_param_array(field, dims)
            elif field == "G":
                # Get the Soil Heat flux if G_form includes the option of
                # Constant G or constant ratio of soil reaching radiation
                if self.G_form[0][0] == TSEB.G_CONSTANT or self.G_form[0][0] == TSEB.G_RATIO:
                    success, self.G_form[1] = self._set_param_array(self.G_form[1], dims)
                # Santanello and Friedls G
                elif self.G_form[0][0] == TSEB.G_TIME_DIFF:
                    # Set the time in the G_form flag to compute the Santanello and
                    # Friedl G
                    self.G_form[1] = self._set_param_array("time", dims)[1]
            elif field == 'S_dn_24':
                success, in_data[field] = self._set_param_array(field, dims)
                if success:
                    self.calc_daily_ET = True
            else:
                # Model specific fields which might need special treatment
                success, inputs = self._set_special_model_input(field, dims)
                if success:
                    in_data.update(inputs)
                else:
                    success, in_data[field] = self._set_param_array(field, dims)

            if not success:
                # Some fields are optional is some circumstances or can be calculated if missing.
                if field in ["SZA", "SAA"]:
                    print("Estimating missing %s parameter" % field)
                    try:
                        in_data['SZA'], in_data['SAA'] = met.calc_sun_angles(temp_data["lat"],
                                                                             temp_data["lon"],
                                                                             temp_data["stdlon"],
                                                                             temp_data["DOY"],
                                                                             temp_data["time"])
                    except KeyError as e:
                        print("ERROR: Cannot calculate or read {}. {} or parameter {} are missing."
                              .format(input_fields[field], field, e))
                        return
                elif field == "p":
                    print("Estimating missing %s parameter" % field)
                    try:
                        in_data["p"] = met.calc_pressure(in_data["alt"])
                    except KeyError as e:
                        print("ERROR: Cannot calculate or read {}. {} or parameter {} are missing."
                              .format(input_fields[field], field, e))
                        return
                elif field == "L_dn":
                    print("Estimating missing %s parameter" % field)
                    try:
                        in_data['L_dn'] = rad.calc_longwave_irradiance(in_data['ea'],
                                                                       in_data['T_A1'],
                                                                       in_data['p'],
                                                                       in_data['z_T'])
                    except KeyError as e:
                        print("ERROR: Cannot calculate or read {}. {} or parameter {} are missing."
                              .format(input_fields[field], field, e))
                        return
                elif (field in ['KN_b', 'KN_c', 'KN_c_dash']
                      and self.resistance_form != TSEB.KUSTAS_NORMAN_1999):
                    print("ERROR: Cannot read {}.".format(input_fields[field]))
                    return
                elif field == "input_mask":
                    print("Please set input_mask=0 for processing the whole image.")
                    return
                elif field == "S_dn_24":
                    print("Provide a valid S_dn_24 (Daily shortwave irradiance) "
                          "value if you want to estimate daily ET")
                elif field in ["alt", "lat", "lon", "stdlon", "DOY", "time"]:
                    print("WARNING!: Non-critical parameter %s "
                          "is invalid or missing..."%field)
                    pass
                else:
                    print('ERROR: file read {}'.format(field))
                    print('Please type a valid filename or a numeric value for '
                          .format(input_fields[field]))
                    return

        temp_data = None

        # ======================================
        # Run the chosen model

        out_data = self.run(in_data, mask)

        # ======================================
        # Save output files

        # Output variables to be saved in images
        all_fields = self._get_output_structure()
        primary_fields = [field for field, save in all_fields.items() if save == S_P]
        ancillary_fields = [field for field, save in all_fields.items() if save == S_A]
        print(primary_fields)
        outdir = dirname(self.p['output_file'])
        if not exists(outdir):
            mkdir(outdir)
        self.write_raster_output(self.p['output_file'], out_data, primary_fields)
        outputfile = splitext(self.p['output_file'])[0] + '_ancillary' + \
                     splitext(self.p['output_file'])[1]
        self.write_raster_output(outputfile, out_data, ancillary_fields)
        print('Saved Files')

        return in_data, out_data

    def process_point_series_array(self):
        ''' Prepare input data and calculate energy fluxes for all the dates in point time-series.

        Parameters
        ----------
        None

        Returns
        -------
        in_data : dict
            All the input data coming into the model.
        out_data : dict
            All the output data coming out of the model.
        '''

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

        # ======================================
        # Process the input

        # Read input data from CSV file
        in_data = pd.read_csv(self.p['input_file'],
                              sep="\s+",
                              index_col=False)
        in_data.index = compose_date(
            years=in_data['year'],
            days=in_data['DOY'],
            hours=in_data['time'],
            minutes=in_data['time'] % 1 * 60)

        # Check if all the required columns are present
        required_columns = self._get_required_data_columns()
        missing = set(required_columns) - (set(in_data.columns))
        if missing:
            print('ERROR: ' + str(list(missing)) + ' not found in file ' + self.p['input_file'])
            return None, None

        # Fill in data fields which might not be in the input file
        if 'SZA' not in in_data.columns:
            sza, _ = met.calc_sun_angles(self.p['lat'], self.p['lon'],
                                         self.p['stdlon'], in_data['DOY'], in_data['time'])
            in_data['SZA'] = sza
        if 'SAA' not in in_data.columns:
            _, saa = met.calc_sun_angles(self.p['lat'], self.p['lon'],
                                         self.p['stdlon'], in_data['DOY'], in_data['time'])
            in_data['SAA'] = saa
        if 'p' not in in_data.columns:
            # Estimate barometric pressure from the altitude if not included in the table
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
        # If longwave irradiance was not provided then estimate it based on air
        # temperature and humidity
        if 'L_dn' not in in_data.columns:
            in_data['L_dn'] = rad.calc_longwave_irradiance(in_data['ea'], in_data['T_A1'],
                                                           in_data['p'], in_data['z_T'])

        # Get the Soil Heat flux if G_form includes the option of measured G
        dims = in_data['LAI'].shape
        if self.G_form[0][0] == TSEB.G_CONSTANT:
            if 'G' in in_data.columns:
                self.G_form[1] = in_data['G']
            else:
                self.G_form[1] = np.ones(dims) * self.G_form[1]
        elif self.G_form[0][0] == TSEB.G_RATIO:
            self.G_form[1] = np.ones(dims) * self.G_form[1]
        elif self.G_form[0][0] == TSEB.G_TIME_DIFF:
            # Set the time in the G_form flag to compute the Santanello and
            # Friedl G
            self.G_form[1] = in_data['time']

        # Set the Kustas and Norman resistance parameters
        if self.resistance_form == 0:
            self.res_params['KN_b'] = np.ones(dims) * self.p['KN_b']
            self.res_params['KN_c'] = np.ones(dims) * self.p['KN_c']
            self.res_params['KN_C_dash'] = np.ones(dims) * self.p['KN_C_dash']

        # ======================================
        # Run the chosen model

        out_data = self.run(in_data.to_records(index=False))
        out_data = pd.DataFrame(data=out_data,
                                index=in_data.index)

        # ======================================
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

    def run(self, in_data, mask=None):
        ''' Execute the routines to calculate energy fluxes.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        mask : int array or None
            If None then fluxes will be calculated for all input points. Otherwise, fluxes will be
            calculated only for points for which mask is 1.

        Returns
        -------
        out_data : dict
            The output data from the model.
        '''

        print("Processing...")

        model_params = {"calcG_params": [self.G_form[0], self.G_form[1]],
                        "resistance_form": [self.resistance_form, self.res_params]}

        if mask is None:
            mask = np.ones(in_data['LAI'].shape, np.int32)

        # Create the output dictionary
        out_data = dict()
        for field in self._get_output_structure():
            out_data[field] = np.zeros(in_data['LAI'].shape, np.float32) + np.nan

        # Esimate diffuse and direct irradiance
        difvis, difnir, fvis, fnir = rad.calc_difuse_ratio(
            in_data['S_dn'], in_data['SZA'], press=in_data['p'])
        out_data['fvis'] = fvis
        out_data['fnir'] = fnir
        out_data['Skyl'] = difvis * fvis + difnir * fnir
        out_data['S_dn_dir'] = in_data['S_dn'] * (1.0 - out_data['Skyl'])
        out_data['S_dn_dif'] = in_data['S_dn'] * out_data['Skyl']

        # ======================================
        # First process bare soil cases

        noVegPixels = in_data['LAI'] <= 0
        noVegPixels = np.logical_or.reduce(
            (in_data['f_c'] <= 0.01,
             in_data['LAI'] <= 0,
             np.isnan(in_data['LAI'])))
        # in_data['LAI'][noVegPixels] = 0
        # in_data['f_c'][noVegPixels] = 0
        i = np.array(np.logical_and(noVegPixels, mask == 1))

        # Calculate roughness
        out_data['z_0M'][i] = in_data['z0_soil'][i]
        out_data['d_0'][i] = 0

        # Net shortwave radition for bare soil
        spectraGrdOSEB = out_data['fvis'] * \
            in_data['rho_vis_S'] + out_data['fnir'] * in_data['rho_nir_S']
        out_data['Sn_S1'][i] = (1. - spectraGrdOSEB[i]) * \
            (out_data['S_dn_dir'][i] + out_data['S_dn_dif'][i])

        # Other fluxes for bare soil
        self._call_flux_model_soil(in_data, out_data, model_params, i)

        # Set canopy fluxes to 0
        out_data['Sn_C1'][i] = 0.0
        out_data['Ln_C1'][i] = 0.0
        out_data['LE_C1'][i] = 0.0
        out_data['H_C1'][i] = 0.0

        # ======================================
        # Then process vegetated cases

        i = np.array(np.logical_and(~noVegPixels, mask == 1))

        # Calculate roughness
        out_data['z_0M'][i], out_data['d_0'][i] = \
            res.calc_roughness(in_data['LAI'][i],
                               in_data['h_C'][i],
                               w_C=in_data['w_C'][i],
                               landcover=in_data['landcover'][i],
                               f_c=in_data['f_c'][i])

        # Net shortwave radiation for vegetation
        F = np.zeros(in_data['LAI'].shape, np.float32)
        F[i] = in_data['LAI'][i] / in_data['f_c'][i]
        # Clumping index
        omega0 = np.zeros(in_data['LAI'].shape, np.float32)
        Omega = np.zeros(in_data['LAI'].shape, np.float32)
        omega0[i] = CI.calc_omega0_Kustas(
            in_data['LAI'][i],
            in_data['f_c'][i],
            x_LAD=in_data['x_LAD'][i],
            isLAIeff=True)
        if self.p['calc_row'][0] == 0:  # randomly placed canopies
            Omega[i] = CI.calc_omega_Kustas(
                omega0[i], in_data['SZA'][i], w_C=in_data['w_C'][i])
        elif self.p['calc_row'][0] == 1:  # row crop canopies
                Omega[i] = CI.calc_omega_rows(in_data['LAI'][i],
                                              in_data['f_c'][i],
                                              theta=in_data['SZA'][i],
                                              psi=self.p['calc_row'][1] - in_data['SAA'][i],
                                              w_c=in_data['w_C'][i],
                                              x_lad=in_data['x_LAD'][i],
                                              is_lai_eff=True)

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

        # Other fluxes for vegetation
        self._call_flux_model_veg(in_data, out_data, model_params, i)

        # Calculate the bulk fluxes
        out_data['LE1'] = out_data['LE_C1'] + out_data['LE_S1']
        out_data['LE_partition'] = out_data['LE_C1'] / out_data['LE1']
        out_data['H1'] = out_data['H_C1'] + out_data['H_S1']
        out_data['R_ns1'] = out_data['Sn_C1'] + out_data['Sn_S1']
        out_data['R_nl1'] = out_data['Ln_C1'] + out_data['Ln_S1']
        out_data['R_n1'] = out_data['R_ns1'] + out_data['R_nl1']
        out_data['delta_R_n1'] = out_data['Sn_C1'] + out_data['Ln_C1']

        if self.water_stress:
            i = mask == 1
            [_, _, _, _, _, _, out_data['LE_0'][i], _,
             out_data['LE_C_0'][i], _, _, _, _, _, _, _, _, _, _] = \
                 pet.shuttleworth_wallace(
                              in_data['T_A1'][i],
                              in_data['u'][i],
                              in_data['ea'][i],
                              in_data['p'][i],
                              out_data['Sn_C1'][i],
                              out_data['Sn_S1'][i],
                              in_data['L_dn'][i],
                              np.maximum(in_data['LAI'][i], 0.01),
                              in_data['h_C'][i],
                              in_data['emis_C'][i],
                              in_data['emis_S'][i],
                              out_data['z_0M'][i],
                              out_data['d_0'][i],
                              in_data['z_u'][i],
                              in_data['z_T'][i],
                              f_c=in_data['f_c'][i],
                              w_C=in_data['w_C'][i],
                              leaf_width=in_data['leaf_width'][i],
                              z0_soil=in_data['z0_soil'][i],
                              x_LAD=in_data['x_LAD'][i],
                              Rst_min=self.p['Rst_min'],
                              R_ss=self.p['R_ss'],
                              calcG_params=[model_params["calcG_params"][0],
                                            model_params["calcG_params"][1][i]],
                              resistance_form=[model_params["resistance_form"][0],
                                               {k: model_params["resistance_form"][1][k][i]
                                                   for k in model_params["resistance_form"][1]}])

            out_data['CWSI'][i] = 1.0 - (out_data['LE_C1'][i] / out_data['LE_C_0'][i])

        if self.calc_daily_ET:
            out_data['ET_day'] = met.flux_2_evaporation(in_data['S_dn_24'] * out_data['LE1'] / in_data['S_dn'],
                                                        t_k=20+273.15,
                                                        time_domain=24)

        print("Finished processing!")
        return out_data

    def _call_flux_model_veg(self, in_data, out_data, model_params, i):
        ''' Call a TSEB_PT model to calculate fluxes for data points containing vegetation.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''

        [out_data['flag'][i], out_data['T_S1'][i], out_data['T_C1'][i],
         out_data['T_AC1'][i], out_data['Ln_S1'][i], out_data['Ln_C1'][i],
         out_data['LE_C1'][i], out_data['H_C1'][i], out_data['LE_S1'][i],
         out_data['H_S1'][i], out_data['G1'][i], out_data['R_S1'][i],
         out_data['R_x1'][i], out_data['R_A1'][i], out_data['u_friction'][i],
         out_data['L'][i], out_data['n_iterations'][i]] = TSEB.TSEB_PT(
            in_data['T_R1'][i],
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
            calcG_params=[model_params["calcG_params"][0],
                          model_params["calcG_params"][1][i]],
            resistance_form=[model_params["resistance_form"][0],
                             {k: model_params["resistance_form"][1][k][i]
                             for k in model_params["resistance_form"][1]}])

    def _call_flux_model_soil(self, in_data, out_data, model_params, i):
        ''' Call a OSEB model to calculate soil fluxes for data points containing no vegetation.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''

        [out_data['flag'][i],
         out_data['Ln_S1'][i],
         out_data['LE_S1'][i],
         out_data['H_S1'][i],
         out_data['G1'][i],
         out_data['R_A1'][i],
         out_data['u_friction'][i],
         out_data['L'][i],
         out_data['n_iterations'][i]] = TSEB.OSEB(in_data['T_R1'][i],
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
                                                  calcG_params=[model_params["calcG_params"][0],
                                                                model_params["calcG_params"][1][i]])

    def _set_param_array(self, parameter, dims, band=1):
        '''Set model input parameter as an array.

        Parameters
        ----------
        parameter : float or string
            If float it should be the value of the parameter. If string of a number it is also
            the value of the parameter. Otherwise it is the name of the parameter and the value, or
            path to raster file which contains the values, is read from the parameters dictionary.
        dims : int list
            The dimensions of the output parameter array.
        band : int (default = 1)
            Band (in GDAL convention) of raster file to be read, if parameter is to be read from a
            raster file.

        Returns
        -------
        success : boolean
            True is the parameter was succefully set, false otherwise.
        array : float array
            The set parameter array.
        '''

        success = True
        array = None

        # See if the parameter is a number
        try:
            array = np.zeros(dims, np.float32) + float(parameter)
            return success, array
        except ValueError:
            pass

        # Otherwise see if the parameter is a parameter name
        try:
            inputString = self.p[parameter]
        except KeyError:
            success = False
            return success, array
        # If it is then get the value of that parameter
        try:
            array = np.zeros(dims, np.float32) + float(inputString)
        except ValueError:
            try:
                fid = gdal.Open(inputString, gdal.GA_ReadOnly)
                if self.subset:
                    array = fid.GetRasterBand(band).ReadAsArray(self.subset[0],
                                                                self.subset[1],
                                                                self.subset[2],
                                                                self.subset[3]).astype(np.float32)
                else:
                    array = fid.GetRasterBand(band).ReadAsArray().astype(np.float32)
            except AttributeError:
                print("%s image not present for parameter %s" % (inputString, parameter))
                success = False
            finally:
                fid = None

        return success, array

    def write_raster_output(self, outfile, output, fields):
        '''Write the specified arrays of a dictionary to a raster file.

        Parameters
        ----------
        outfile : string
            Path to the output raster. If the path ends in ".nc" the output will be saved in a
            netCDF file. If the path ends in ".vrt" then the outputs will be saved in a GDAL
            virtual raster with the actual data saved as GeoTIFFs (one per field) in .data
            sub-folder. Otherwise, the output will be saved as one GeoTIFF.
        output : dict
            The dictionary containing the output data arrays.
        fields : string list
            The list of output fields from the output dictionary to save to file.

        Returns
        -------
        None
        '''

        # If the output file has .nc extension then save it as netCDF,
        # otherwise assume that the output should be a GeoTIFF
        ext = splitext(outfile)[1]
        if ext.lower() == ".nc":
            driver_name = "netCDF"
            opt = ["FORMAT=NC4"]
            opt = []
        elif ext.lower() == ".vrt":
            driver_name = "VRT"
            opt = []
        else:
            driver_name = "COG"
            opt = ['COMPRESS=DEFLATE', 'PREDICTOR=YES', 'BIGTIFF=IF_SAFER']
        if driver_name in ["COG", "netCDF"]:
            # Save the data using GDAL by first creating a MEM layer and later using Translate
            rows, cols = np.shape(output['H1'])
            driver = gdal.GetDriverByName("MEM")
            nbands = len(fields)
            ds = driver.Create("MEM", cols, rows, nbands, gdal.GDT_Float32)
            ds.SetGeoTransform(self.geo)
            ds.SetProjection(self.prj)
            for i, field in enumerate(fields):
                band = ds.GetRasterBand(i + 1)
                band.WriteArray(output[field])
                band.SetStatistics(*band.ComputeStatistics(0))
            out_ds = gdal.Translate(outfile, ds, format=driver_name, creationOptions=opt,
                                    noData=None)
            # If GDAL drivers for other formats do not exist then default to GeoTiff
            if out_ds is None:
                print("Warning: Selected GDAL driver is not supported! Saving as GeoTiff!")
                driver_name = "GTiff"
                opt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']
                gdal.Translate(outfile, ds, format=driver_name, creationOptions=opt, noData=None)
            out_ds = None
            ds = None
            # In case of netCDF format use netCDF4 module to assign proper names
            # to variables (GDAL can't do this). Also it seems that GDAL has
            # problems assigning projection to all the bands so fix that.
            if driver_name == "netCDF":
                ds = Dataset(outfile, 'a')
                grid_mapping = ds["Band1"].grid_mapping
                for i, field in enumerate(fields):
                    ds.renameVariable('Band{}'.format(i + 1), field)
                    ds[field].grid_mapping = grid_mapping
                ds.close()

        else:
            # Save each individual oputput in a GeoTIFF file in .data directory using GDAL
            out_dir = join(dirname(outfile),
                           splitext(basename(outfile))[0] + ".data")
            if not exists(out_dir):
                mkdir(out_dir)
            out_files = []
            rows, cols = np.shape(output['H1'])
            outfile_tif = (splitext(basename(outfile))[0]).replace("_ancillary", "")
            for i, field in enumerate(fields):
                driver = gdal.GetDriverByName("MEM")
                out_path = join(out_dir, f"{outfile_tif}_{field}.tif")
                ds = driver.Create("MEM", cols, rows, 1, gdal.GDT_Float32)
                ds.SetGeoTransform(self.geo)
                ds.SetProjection(self.prj)
                band = ds.GetRasterBand(1)
                band.WriteArray(output[field])
                opt = ['COMPRESS=DEFLATE', 'PREDICTOR=YES', 'BIGTIFF=IF_SAFER']
                out_ds = gdal.Translate(out_path, ds, format="COG", creationOptions=opt,
                                        noData=None, stats=True)
                # If GDAL drivers for other formats do not exist then default to GeoTiff
                if out_ds is None:
                    opt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']
                    gdal.Translate(out_path, ds, format="GTiff", creationOptions=opt, noData=None,
                                   stats=True)
                out_ds = None
                ds = None
                out_files.extend([out_path])

            # Create the Virtual Raster Table
            out_vrt = out_dir.replace('.data', '.vrt')
            print(out_files)
            gdal.BuildVRT(out_vrt, out_files, separate=True)

    def _get_output_structure(self):
        ''' Output fields' names for TSEB model.

        Parameters
        ----------
        None

        Returns
        -------
        output_structure: ordered dict
            Names of the output fields as keys and instructions on whether the output
            should be saved to file as values.
        '''

        output_structure = OrderedDict([
            # Energy fluxes
            ('R_n1', S_P),   # net radiation reaching the surface at time t1
            ('R_ns1', S_A),  # net shortwave radiation reaching the surface at time t1
            ('R_nl1', S_A),  # net longwave radiation reaching the surface at time t1
            ('delta_R_n1', S_A),  # net radiation divergence in the canopy at time t1
            ('Sn_S1', S_N),  # Shortwave radiation reaching the soil at time t1
            ('Sn_C1', S_N),  # Shortwave radiation intercepted by the canopy at time t1
            ('Ln_S1', S_N),  # Longwave radiation reaching the soil at time t1
            ('Ln_C1', S_N),  # Longwave radiation intercepted by the canopy at time t1
            ('H_C1', S_A),  # canopy sensible heat flux (W/m^2) at time t1
            ('H_S1', S_N),  # soil sensible heat flux (W/m^2) at time t1
            ('H1', S_P),  # total sensible heat flux (W/m^2) at time t1
            ('LE_C1', S_A),  # canopy latent heat flux (W/m^2) at time t1
            ('LE_S1', S_N),  # soil latent heat flux (W/m^2) at time t1
            ('LE1', S_P),  # total latent heat flux (W/m^2) at time t1
            ('LE_partition', S_A),  # Latent Heat Flux Partition (LEc/LE) at time t1
            ('G1', S_P),  # ground heat flux (W/m^2) at time t1
            # temperatures (might not be accurate)
            ('T_C1', S_A),  # canopy temperature at time t1 (deg C)
            ('T_S1', S_A),  # soil temperature at time t1 (deg C)
            ('T_AC1', S_N),  # air temperature at the canopy interface at time t1 (deg C)
            # resistances
            # resistance to heat transport in the surface layer (s/m) at time t1
            ('R_A1', S_A),
            # resistance to heat transport in the canopy surface layer (s/m) at time t1
            ('R_x1', S_A),
            # resistance to heat transport from the soil surface (s/m) at time t1 fluxes
            ('R_S1', S_A),
            # miscaleneous
            ('albedo1', S_N),    # surface albedo (Rs_out/Rs_in)
            ('omega0', S_N),  # nadir view vegetation clumping factor
            ('alpha', S_N),  # the priestly Taylor factor
            ('Ri', S_N),  # Richardson number at time t1
            ('L', S_A),  # Monin Obukhov Length at time t1
            ('u_friction', S_A),  # Friction velocity
            ('theta_s1', S_N),  # Sun zenith angle at time t1
            ('F', S_N),  # Leaf Area Index
            ('z_0M', S_N),  # Aerodynamic roughness length for momentum trasport (m)
            ('d_0', S_N),  # Zero-plane displacement height (m)
            ('Skyl', S_N),
            ('flag', S_A),  # Quality flag
            ('n_iterations', S_N)])  # Number of iterations before model converged to stable value


        if self.calc_daily_ET:
            output_structure['ET_day'] = S_P

        if self.water_stress:
            output_structure['LE_0'] = S_A
            output_structure['LE_C_0'] = S_A
            output_structure['CWSI'] = S_P

        return output_structure

    def _get_input_structure(self):
        ''' Input fields' names for TSEB_PT model. Only relevant for image processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        input_fields: string ordered dict
            Names (keys) and descriptions (values) of TSEB_PT input fields.
        '''

        input_fields = OrderedDict([
                            # General parameters
                            ("T_R1", "Land Surface Temperature"),
                            ("LAI", "Leaf Area Index"),
                            ("VZA", "View Zenith Angle for LST"),
                            ("landcover", "Landcover"),
                            ("input_mask", "Input Mask"),
                            # Vegetation parameters
                            ("f_c", "Fractional Cover"),
                            ("h_C", "Canopy Height"),
                            ("w_C", "Canopy Width Ratio"),
                            ("f_g", "Green Vegetation Fraction"),
                            ("leaf_width", "Leaf Width"),
                            ("x_LAD", "Leaf Angle Distribution"),
                            ("alpha_PT", "Initial Priestley-Taylor Alpha Value"),
                            # Spectral Properties
                            ("rho_vis_C", "Leaf PAR Reflectance"),
                            ("tau_vis_C", "Leaf PAR Transmitance"),
                            ("rho_nir_C", "Leaf NIR Reflectance"),
                            ("tau_nir_C", "Leaf NIR Transmitance"),
                            ("rho_vis_S", "Soil PAR Reflectance"),
                            ("rho_nir_S", "Soil NIR Reflectance"),
                            ("emis_C", "Leaf Emissivity"),
                            ("emis_S", "Soil Emissivity"),
                            # Illumination conditions
                            ("lat", "Latitude"),
                            ("lon", "Longitude"),
                            ("stdlon", "Standard Longitude"),
                            ("time", "Observation Time for LST"),
                            ("DOY", "Observation Day Of Year for LST"),
                            ("SZA", "Sun Zenith Angle"),
                            ("SAA", "Sun Azimuth Angle"),
                            # Meteorological parameters
                            ("T_A1", "Air temperature"),
                            ("u", "Wind Speed"),
                            ("ea", "Vapour Pressure"),
                            ("alt", "Altitude"),
                            ("p", "Pressure"),
                            ("S_dn", "Shortwave Irradiance"),
                            ("z_T", "Air Temperature Height"),
                            ("z_u", "Wind Speed Height"),
                            ("z0_soil", "Soil Roughness"),
                            ("L_dn", "Longwave Irradiance"),
                            # Resistance parameters
                            ("KN_b", "Kustas and Norman Resistance Parameter b"),
                            ("KN_c", "Kustas and Norman Resistance Parameter c"),
                            ("KN_C_dash", "Kustas and Norman Resistance Parameter c-dash"),
                            # Soil heat flux parameter
                            ("G", "Soil Heat Flux Parameter"),
                            ('S_dn_24', 'Daily shortwave irradiance')])

        return input_fields

    def _set_special_model_input(self, field, dims):
        ''' Special processing for setting certain input fields. Only relevant for image processing
        mode.

        Parameters
        ----------
        field : string
            The name of the input field for which the special processing is needed.
        dims : int list
            The dimensions of the output parameter array.

        Returns
        -------
        success : boolean
            True is the parameter was succefully set, false otherwise.
        inputs : dict
            Dictionary with keys holding input field name and value holding the input value.
        '''
        return False, None

    def _get_required_data_columns(self):
        ''' Input columns' names required in an input asci table for TSEB_PT model. Only relevant
        for point time-series processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        required_columns : string tuple
            Names of the required input columns.
        '''

        required_columns = ('year',
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
        return required_columns

    def _get_subset(self, roi_shape, raster_proj_wkt, raster_geo_transform):

        # Find extent of ROI in roiShape projection
        roi = ogr.Open(roi_shape)
        roi_layer = roi.GetLayer()
        roi_extent = roi_layer.GetExtent()

        # Convert the extent to raster projection
        roi_proj = roi_layer.GetSpatialRef()
        raster_proj = osr.SpatialReference()
        raster_proj.ImportFromWkt(raster_proj_wkt)
        try:
            # For GDAL 3 indicate the "legacy" axis order
            # https://gdal.org/tutorials/osr_api_tut.html#crs-and-axis-order
            # https://github.com/OSGeo/gdal/issues/1546
            roi_proj = roi_proj.Clone()
            roi_proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            raster_proj = raster_proj.Clone()
            raster_proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            # For GDAL 2 do nothing
            pass
        transform = osr.CoordinateTransformation(roi_proj, raster_proj)
        point_UL = ogr.CreateGeometryFromWkt("POINT ({} {})"
                                             .format(min(roi_extent[0], roi_extent[1]),
                                                     max(roi_extent[2], roi_extent[3])))
        point_UL.Transform(transform)
        point_UL = point_UL.GetPoint()
        point_LR = ogr.CreateGeometryFromWkt("POINT ({} {})"
                                             .format(max(roi_extent[0], roi_extent[1]),
                                                     min(roi_extent[2], roi_extent[3])))
        point_LR.Transform(transform)
        point_LR = point_LR.GetPoint()

        # Get pixel location of this extent
        ulX = raster_geo_transform[0]
        ulY = raster_geo_transform[3]
        pixel_size = raster_geo_transform[1]
        pixel_UL = [max(int(math.floor((ulY - point_UL[1]) / pixel_size)), 0),
                    max(int(math.floor((point_UL[0] - ulX) / pixel_size)), 0)]
        pixel_LR = [int(round((ulY - point_LR[1]) / pixel_size)),
                    int(round((point_LR[0] - ulX) / pixel_size))]

        # Get projected extent
        point_proj_UL = (ulX + pixel_UL[1] * pixel_size, ulY - pixel_UL[0] * pixel_size)

        # Convert to xoff, yoff, xcount, ycount as required by GDAL ReadAsArray()
        subset_pix = [pixel_UL[1], pixel_UL[0],
                      pixel_LR[1] - pixel_UL[1], pixel_LR[0] - pixel_UL[0]]

        # Get the geo transform of the subset
        subset_geo_transform = [point_proj_UL[0], pixel_size, raster_geo_transform[2],
                                point_proj_UL[1], raster_geo_transform[4], -pixel_size]

        return subset_pix, subset_geo_transform


class PyDTD(PyTSEB):

    def __init__(self, parameters):
        super().__init__(parameters)

    def _get_input_structure(self):
        ''' Input fields' names for DTD model.  Only relevant for image processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        outputStructure: string ordered dict
            Names (keys) and descriptions (values) of DTD input fields.
        '''

        input_fields = super()._get_input_structure()
        input_fields["T_R0"] = "Early Morning Land Surface Temperature"
        input_fields["T_A0"] = "Early Morning Air Temperature"
        return input_fields

    def _get_required_data_columns(self):
        ''' Input columns' names required in an input asci table for TSEB_PT model. Only relevant
        for point time-series processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        required_columns : string tuple
            Names of the required input columns.
        '''

        required_columns = ('year',
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

        return required_columns

    def _call_flux_model_veg(self, in_data, out_data, model_params, i):
        ''' Call a DTD model to calculate fluxes for data points containing vegetation.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''

        [out_data['flag'][i], out_data['T_S1'][i], out_data['T_C1'][i],
            out_data['T_AC1'][i], out_data['Ln_S1'][i], out_data['Ln_C1'][i],
            out_data['LE_C1'][i], out_data['H_C1'][i], out_data['LE_S1'][i],
            out_data['H_S1'][i], out_data['G1'][i], out_data['R_S1'][i],
            out_data['R_x1'][i], out_data['R_A1'][i], out_data['u_friction'][i],
            out_data['L'][i], out_data['Ri'], out_data['n_iterations'][i]] = TSEB.DTD(
                in_data['T_R0'][i],
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
                calcG_params=[model_params["calcG_params"][0],
                              model_params["calcG_params"][1][i]],
                resistance_form=[model_params["resistance_form"][0],
                                 {k: model_params["resistance_form"][1][k][i]
                                  for k in model_params["resistance_form"][1]}])

    def _call_flux_model_soil(self, in_data, out_data, model_params, i):
        ''' Call a OSEB model to calculate soil fluxes for data points containing no vegetation.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''

        [out_data['flag'][i],
         out_data['Ln_S1'][i],
         out_data['LE_S1'][i],
         out_data['H_S1'][i],
         out_data['G1'][i],
         out_data['R_A1'][i],
         out_data['u_friction'][i],
         out_data['L'][i],
         out_data['n_iterations'][i]] = TSEB.OSEB(in_data['T_R1'][i],
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
                                                  calcG_params=[model_params["calcG_params"][0],
                                                                model_params["calcG_params"][1][i]],
                                                  T0_K=(in_data['T_R0'][i], in_data['T_A0'][i]))


class PyTSEB2T(PyTSEB):

    def __init__(self, parameters):
        super().__init__(parameters)

    def _get_input_structure(self):
        ''' Input fields' names for TSEB_2T model.  Only relevant for image processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        outputStructure: string ordered dict
            Names (keys) and descriptions (values) of TSEB_2T input fields.
        '''

        input_fields = super()._get_input_structure()
        del input_fields["T_R1"]
        input_fields["T_C"] = "Canopy Temperature"
        input_fields["T_S"] = "Soil Temperature"
        return input_fields

    def _set_special_model_input(self, field, dims):
        ''' Special processing for setting certain input fields. Only relevant for image processing
        mode.

        Parameters
        ----------
        field : string
            The name of the input field for which the special processing is needed.
        dims : int list
            The dimensions of the output parameter array.

        Returns
        -------
        success : boolean
            True is the parameter was succefully set, false otherwise.
        array : float array
            The set parameter array.
        '''

        if field == "T_C":
            success, val = self._set_param_array("T_R1", dims)
            inputs = {field: val}
        elif field == "T_S":
            success, val = self._set_param_array("T_R1", dims, band=2)
            inputs = {field: val}
        else:
            success = False
            inputs = None
        return success, inputs

    def _get_required_data_columns(self):
        ''' Input columns' names required in an input asci table for TSEB_PT model. Only relevant
        for point time-series processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        required_columns : string tuple
            Names of the required input columns.
        '''

        required_columns = ('year',
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
        return required_columns

    def _call_flux_model_veg(self, in_data, out_data, model_params, i):
        ''' Call a TSEB_2T model to calculate fluxes for data points containing vegetation.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''

        [out_data['flag'][i], out_data['T_AC1'][i], out_data['Ln_S1'][i],
         out_data['Ln_C1'][i], out_data['LE_C1'][i], out_data['H_C1'][i],
         out_data['LE_S1'][i], out_data['H_S1'][i], out_data['G1'][i],
         out_data['R_S1'][i], out_data['R_x1'][i], out_data['R_A1'][i],
         out_data['u_friction'][i], out_data['L'][i], out_data['n_iterations'][i]] = TSEB.TSEB_2T(
             in_data['T_C'][i],
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
             calcG_params=[model_params["calcG_params"][0],
                           model_params["calcG_params"][1][i]],
             resistance_form=[model_params["resistance_form"][0],
                              {k: model_params["resistance_form"][1][k][i]
                              for k in model_params["resistance_form"][1]}])

    def _call_flux_model_soil(self, in_data, out_data, model_params, i):
        ''' Call a OSEB model to calculate soil fluxes for data points containing no vegetation.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''

        [out_data['flag'][i],
         out_data['Ln_S1'][i],
         out_data['LE_S1'][i],
         out_data['H_S1'][i],
         out_data['G1'][i],
         out_data['R_A1'][i],
         out_data['u_friction'][i],
         out_data['L'][i],
         out_data['n_iterations'][i]] = TSEB.OSEB(in_data['T_S'][i],
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
                                                  calcG_params=[model_params["calcG_params"][0],
                                                                model_params["calcG_params"][1][i]])


class PydisTSEB(PyTSEB):

    def __init__(self, parameters):

        super().__init__(parameters)
        # Method for ensuring consistency between low- and high-resolution fluxes
        self.flux_LR_method = self.p["flux_LR_method"]
        # Correct LST (if True) or air temperature (if false) during disaggregation.
        self.correct_LST = self.p["correct_LST"]

    def _get_input_structure(self):
        ''' Input fields' names for disTSEB model.  Only relevant for image processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        outputStructure: string ordered dict
            Names (keys) and descriptions (values) of disTSEB input fields.
        '''

        input_fields = super()._get_input_structure()
        input_fields["flux_LR"] = "pyTSEB Low Resolution Flux date"
        input_fields["flux_LR_ancillary"] = "pyTSEB Low Resolution ancillary Flux data"

        return input_fields

    def _get_output_structure(self):
        ''' Input fields' names for disTSEB model.  Only relevant for image processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        outputStructure: string ordered dict
            Names (keys) and descriptions (values) of disTSEB input fields.
        '''

        output_structure = super()._get_output_structure()
        output_structure["T_offset"] = S_A
        output_structure["T_offset_orig"] = S_A
        output_structure["counter"] = S_A
        return output_structure

    def _set_special_model_input(self, field, dims):
        ''' Special processing for setting certain input fields. Only relevant for image processing
        mode.

        Parameters
        ----------
        field : string
            The name of the input field for which the special processing is needed.
        dims : int list
            The dimensions of the output parameter array.

        Returns
        -------
        success : boolean
            True is the parameter was succefully set, false otherwise.
        array : float array
            The set parameter array.
        '''
        if field in ["flux_LR", "flux_LR_ancillary"]:
            # Low resolution data in case disaggregation is to be used.
            inputs = {}
            fid = gdal.Open(self.p[field], gdal.GA_ReadOnly)
            if fid is None:
                print("ERROR: Low resolution data for disaggregation is not avaiable.")
                return False, None
            prj_LR = fid.GetProjection()
            geo_LR = fid.GetGeoTransform()
            subset = []
            if "subset" in self.p:
                subset, geo_LR = self._get_subset(self.p["subset"], prj_LR, geo_LR)
                inputs[field] = fid.GetRasterBand(1).ReadAsArray(subset[0],
                                                                 subset[1],
                                                                 subset[2],
                                                                 subset[3]).astype(np.float32)
            else:
                inputs[field] = fid.GetRasterBand(1).ReadAsArray().astype(np.float32)
            inputs['scale'] = [geo_LR, prj_LR, self.geo, self.prj]
            success = True
        else:
            success = False
            inputs = None

        return success, inputs

    def _call_flux_model_soil(self, in_data, out_data, model_params, i):
        return

    def _call_flux_model_veg(self, in_data, out_data, model_params, i):
        ''' Call a dis_TSEB model to calculate fluxes

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''

        print('Running dis TSEB for the whole image')

        [out_data['flag'],
         out_data['T_S1'],
         out_data['T_C1'],
         out_data['T_AC1'],
         out_data['Ln_S1'],
         out_data['Ln_C1'],
         out_data['LE_C1'],
         out_data['H_C1'],
         out_data['LE_S1'],
         out_data['H_S1'],
         out_data['G1'],
         out_data['R_S1'],
         out_data['R_x1'],
         out_data['R_A1'],
         out_data['u_friction'],
         out_data['L'],
         out_data['n_iterations'],
         out_data['T_offset'],
         out_data['counter'],
         out_data['T_offset_orig']] = dis_TSEB.dis_TSEB(
             in_data['flux_LR'],
             in_data['scale'],
             in_data['T_R1'],
             in_data['VZA'],
             in_data['T_A1'],
             in_data['u'],
             in_data['ea'],
             in_data['p'],
             out_data['Sn_C1'],
             out_data['Sn_S1'],
             in_data['L_dn'],
             in_data['LAI'],
             in_data['h_C'],
             in_data['emis_C'],
             in_data['emis_S'],
             out_data['z_0M'],
             out_data['d_0'],
             in_data['z_u'],
             in_data['z_T'],
             UseL=in_data['flux_LR_ancillary'],
             f_c=in_data['f_c'],
             f_g=in_data['f_g'],
             w_C=in_data['w_C'],
             leaf_width=in_data['leaf_width'],
             z0_soil=in_data['z0_soil'],
             alpha_PT=in_data['alpha_PT'],
             x_LAD=in_data['x_LAD'],
             calcG_params=model_params["calcG_params"],
             resistance_form=model_params["resistance_form"],
             flux_LR_method=self.flux_LR_method,
             correct_LST=self.correct_LST)
