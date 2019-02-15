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

from re import match

from pyTSEB.PyTSEB import PyTSEB, PyTSEB2T, PyDTD, PydisTSEB


class TSEBConfigFileInterface():

    def __init__(self):

        # Variables common to both image and point series runs
        self.input_site_description_vars = (
            'landcover',
            'lat',
            'lon',
            'alt',
            'stdlon',
            'z_T',
            'z_u',
            'z0_soil')
        self.input_vegetation_properties_vars = (
            'leaf_width',
            'alpha_PT',
            'x_LAD')
        self.input_spectral_properties_vars = (
            'emis_C',
            'emis_S',
            'rho_vis_C',
            'tau_vis_C',
            'rho_nir_C',
            'tau_nir_C',
            'rho_vis_S',
            'rho_nir_S')
        self.input_model_formulation_vars = (
            'model',
            'resistance_form',
            'KN_b',
            'KN_c',
            'KN_C_dash',
            'G_form',
            'G_constant',
            'G_ratio',
            'G_amp',
            'G_phase',
            'G_shape',
            'calc_row',
            'row_az',
            'output_file',
            'correct_LST',
            'flux_LR_method')

        # Variables only for image runs
        self.input_image_vars = (
            'T_R1',
            'T_R0',
            'VZA',
            'LAI',
            'f_c',
            'f_g',
            'h_C',
            'w_C',
            'input_mask',
            'subset',
            'time',
            'DOY',
            'T_A1',
            'T_A0',
            'u',
            'ea',
            'S_dn',
            'L_dn',
            'p',
            'flux_LR',
            'flux_LR_ancillary')

        # variables only for point series runs
        self.input_point_vars = (
            'input_file',
            'f_c',
            'f_g',
            'w_C')

        self.params = {}
        self.ready = False

    def parse_input_config(self, input_file, is_image=False):
        ''' Parses the information contained in a configuration file into a dictionary'''

        # Prepare a list of expected input variables
        if is_image:
            input_vars = list(self.input_image_vars)
        else:
            input_vars = list(self.input_point_vars)
        input_vars.extend(self.input_site_description_vars)
        input_vars.extend(self.input_spectral_properties_vars)
        input_vars.extend(self.input_vegetation_properties_vars)
        input_vars.extend(self.input_model_formulation_vars)

        # Read contents of the configuration file
        config_data = dict()
        try:
            with open(input_file, 'r') as fid:
                for line in fid:
                    if match('\s', line):  # skip empty line
                        continue
                    elif match('#', line):  # skip comment line
                        continue
                    elif '=' in line:
                        # Remove comments in case they exist
                        line = line.split('#')[0].rstrip(' \r\n')
                        field, value = line.split('=')
                        if field in input_vars:
                            config_data[field] = value
        except IOError:
            print('Error reading ' + input_file + ' file')

        return config_data

    def get_data(self, config_data, is_image):
        '''Parses the parameters in a configuration file directly to TSEB variables for running
           TSEB'''
        try:

            for var_name in self.input_site_description_vars:
                if is_image:
                    self.params[var_name] = str(config_data[var_name]).strip('"')
                else:
                    self.params[var_name] = float(config_data[var_name])

            for var_name in self.input_vegetation_properties_vars:
                if is_image:
                    self.params[var_name] = str(config_data[var_name]).strip('"')
                else:
                    self.params[var_name] = float(config_data[var_name])

            for var_name in self.input_spectral_properties_vars:
                if is_image:
                    self.params[var_name] = str(config_data[var_name]).strip('"')
                else:
                    self.params[var_name] = float(config_data[var_name])

            for var_name in self.input_model_formulation_vars:
                if var_name in ["model", "output_file"]:
                    self.params[var_name] = str(config_data[var_name]).strip('"')
                elif var_name == "resistance_form":
                    self.params[var_name] = int(config_data[var_name])
                elif var_name == "calc_row":
                    if 'calc_row' not in config_data or int(config_data['calc_row']) == 0:
                        self.params['calc_row'] = [0, 0]
                    else:
                        self.params['calc_row'] = [1, float(config_data['row_az'])]
                elif var_name == "G_form":
                    if int(config_data['G_form']) == 0:
                        self.params['G_form'] = [[0], float(config_data['G_constant'])]
                    elif int(config_data['G_form']) == 1:
                        self.params['G_form'] = [[1], float(config_data['G_ratio'])]
                    elif int(config_data['G_form']) == 2:
                        self.params['G_form'] = [[2,
                                                  float(config_data['G_amp']),
                                                  float(config_data['G_phase']),
                                                  float(config_data['G_shape'])],
                                                 12.0]
                # disTSEB specific parameters can be ignored by other models.
                elif var_name in ["flux_LR_method", "correct_LST"]:
                    if self.params["model"] == "disTSEB":
                        self.params[var_name] = str(config_data[var_name]).strip('"')
                elif var_name in ["row_az", "G_constant", "G_ratio", "G_amp", "G_phase",
                                  "G_shape"]:
                    pass
                else:
                    if is_image:
                        self.params[var_name] = str(config_data[var_name]).strip('"')
                    else:
                        self.params[var_name] = float(config_data[var_name])

            if is_image:
                # Get the input parameters which are specific for running in image
                # mode
                for var in self.input_image_vars:
                    try:
                        self.params[var] = str(config_data[var]).strip('"')
                    except KeyError as e:
                        if (var == 'T_A0' or var == 'T_R0') and self.params['model'] != 'DTD':
                            pass
                        elif (var == 'subset'):
                            pass
                        elif (var == 'flux_LR' or var == 'flux_LR_ancillary') and\
                             self.params['model'] != 'disTSEB':
                            pass
                        else:
                            raise e
            else:
                # Get the input parameters which are specific for running in point
                # series mode
                self.params['input_file'] = str(config_data['input_file']).strip('"')
                self.params['f_c'] = float(config_data['f_c'])
                self.params['f_g'] = float(config_data['f_g'])
                self.params['w_C'] = float(config_data['w_C'])

            self.ready = True

        except KeyError as e:
            print('Error: missing parameter '+str(e)+' in the input data.')
        except ValueError as e:
            print('Error: '+str(e))

    def run(self, is_image):

        if self.ready:
            if self.params['model'] == "TSEB_PT":
                model = PyTSEB(self.params)
            elif self.params['model'] == "TSEB_2T":
                model = PyTSEB2T(self.params)
            elif self.params['model'] == "DTD":
                model = PyDTD(self.params)
            elif self.params['model'] == "disTSEB":
                model = PydisTSEB(self.params)
            else:
                print("Unknown model: " + self.params['model'] + "!")
                return None
            if is_image:
                model.process_local_image()
            else:
                in_data, out_data = model.process_point_series_array()
                return in_data, out_data
        else:
            print("pyTSEB will not be run due to errors in the input data.")
