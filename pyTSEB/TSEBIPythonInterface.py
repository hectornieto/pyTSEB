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

import sys

import ipywidgets as widgets
from IPython.display import display

from .TSEBConfigFileInterface import TSEBConfigFileInterface


class TSEBIPythonInterface(TSEBConfigFileInterface):

    def __init__(self):

        TSEBConfigFileInterface.__init__(self)

        '''Initialize input variables  with default  values'''
        self.input_file = './Input/ExampleTableInput.txt'
        self.output_text_file = './Output/test.txt'
        self.output_image_file = './Output/test.tif'

        # MOdel to run
        self.model = 'TSEB_PT'
        # Site Description
        self.lat = 36.95
        self.lon = 2.33
        self.alt = 200
        self.stdlon = 15
        self.zu = 2.5
        self.zt = 3.5
        # Spectral Properties
        self.rho_vis_C = 0.07
        self.rho_nir_C = 0.32
        self.tau_vis_C = 0.08
        self.tau_nir_C = 0.33
        self.rho_vis_S = 0.15
        self.rho_nir_S = 0.25
        self.emis_C = 0.98
        self.emis_S = 0.95
        # Surface Properties
        self.max_PT = 1.26
        self.x_LAD = 1.0
        self.leaf_width = 0.1
        self.z0soil = 0.01
        self.landcover = 12
        # Resistance options
        self.res = 0
        self.KN_b = 0.012
        self.KN_c = 0.0038
        self.KN_C_dash = 90.0
        # RowCrop calculation
        self.row = False
        self.row_az = 90
        # Soil Heat Flux calculation
        self.G_form = 1
        self.Gconstant = 0
        self.Gratio = 0.35
        self.G_amp = 0.35
        self.G_phase = 3
        self.G_shape = 24
        # Default Vegetation variables
        self.f_c = 1.0
        self.f_g = 1.0
        self.w_c = 1.0
        # Output variables saved in images
        # self.fields=('H1','LE1','R_n1','G1')
        # Ancillary output variables
        #self.anc_fields=('H_C1','LE_C1','LE_partition','T_C1', 'T_S1','R_ns1','R_nl1', 'u_friction', 'L')
        # File Configuration variables

    def point_time_series_widget(self):
        '''Creates a jupyter notebook GUI for running TSEB for a point time series dataset'''

        # Load and save configuration buttons
        self.w_loadconfig = widgets.Button(
            description='Load Configuration File')
        self.w_saveconfig = widgets.Button(
            description='Save Configuration File')
        # Input and output ascii files
        self.w_input = widgets.Button(description='Select Input File')
        self.w_inputtxt = widgets.Text(
            description='Input File :',
            value=self.input_file,
            width=500)
        self.w_output = widgets.Button(description='Select Output File')
        self.w_outputtxt = widgets.Text(
            description='Output File :',
            value=self.output_text_file,
            width=500)
        # Run pyTSEB button
        self.w_runmodel = widgets.Button(
            description='Run pyTSEB',
            background_color='green')
        # Create TSEB options widgets
        self.select_model()
        self.define_site_description_time_series()
        self.spectral_properties_time_series()
        self.surface_properties_time_series()
        self.resistances_time_series()
        self.additional_options_point()
        # Model Selection and create tabs
        tabs = widgets.Tab(
            children=[
                self.w_model,
                self.site_page,
                self.spec_page,
                self.veg_page,
                self.res_page,
                self.opt_page])
        tabs.set_title(0, 'TSEB Model')
        tabs.set_title(1, 'Site Description')
        tabs.set_title(2, 'Spectral Properties')
        tabs.set_title(3, 'Canopy Description')
        tabs.set_title(4, 'Resistance Model')
        tabs.set_title(5, 'Additional Options')
        # Display widgets
        display(self.w_loadconfig)
        display(widgets.HBox([self.w_input, self.w_inputtxt]))
        display(widgets.HBox([self.w_output, self.w_outputtxt]))
        display(tabs)
        display(self.w_saveconfig)
        display(self.w_runmodel)
        # Handle interactions
        self.w_res.on_trait_change(self._on_res_change, 'value')
        self.w_row.on_trait_change(self._on_row_change, 'value')
        self.w_G_form.on_trait_change(self._on_G_change, 'value')
        self.w_input.on_click(
            lambda b: self._on_input_clicked(b, 'Input Text', self.w_inputtxt))
        self.w_output.on_click(self._on_output_clicked)
        self.w_loadconfig.on_click(self._on_loadconfig_clicked)
        self.w_saveconfig.on_click(self._on_saveconfig_clicked)
        self.w_runmodel.on_click(self._on_runmodel_clicked)

        self.is_image = False

    def local_image_widget(self):
        '''Creates a jupyter notebook GUI for running TSEB for an image'''

        # Load and save configuration buttons
        self.w_loadconfig = widgets.Button(
            description='Load Configuration File')
        self.w_saveconfig = widgets.Button(
            description='Save Configuration File')
        # Input and output images
        self.w_T_R1_But = widgets.Button(
            description='Browse Radiometric Surface Temperature Image')
        self.w_T_R1 = widgets.Text(description='(K):', value='0', width=500)
        self.w_T_R0_But = widgets.Button(
            description='Browse Sunrise Radiometric Surface Temperature Image')
        self.w_T_R0 = widgets.Text(description='(K):', value='0', width=500)
        self.w_T_R0_But.visible = False
        self.w_T_R0.visible = False
        self.w_VZA = widgets.Button(description='Browse VZA Image')
        self.w_VZAtxt = widgets.Text(description='VZA:', value='0', width=500)
        self.w_LAI = widgets.Button(description='Browse LAI Image')
        self.w_LAItxt = widgets.Text(description='LAI:', value='1', width=500)
        self.w_Hc = widgets.Button(description='Browse H canopy Image')
        self.w_Hctxt = widgets.Text(
            description='H canopy:', value='1', width=500)
        self.w_w_C = widgets.Button(
            description='Browse Canopy widht/height Image')
        self.w_w_Ctxt = widgets.Text(
            description='w_C canopy:', value=str(
                self.w_c), width=500)
        self.w_f_c = widgets.Button(description='Browse F cover Image')
        self.w_f_ctxt = widgets.Text(
            description='F cover:', value=str(
                self.f_c), width=500)
        self.w_f_g = widgets.Button(description='Browse F green Image')
        self.w_f_gtxt = widgets.Text(
            description='F green:', value=str(
                self.f_g), width=500)
        self.w_mask = widgets.Button(description='Browse Image Mask')
        self.w_masktxt = widgets.Text(
            description='Mask:', value='0', width=500)
        self.w_output = widgets.Button(description='Select Output File')
        self.w_outputtxt = widgets.Text(
            description='Output File :',
            value=self.output_image_file,
            width=500)
        # Run pyTSEB button
        self.w_runmodel = widgets.Button(
            description='Run pyTSEB',
            background_color='green')
        # Create TSEB options widgets
        self.select_model()
        self.define_site_description_image()
        self.meteorology()
        self.spectral_properties_image()
        self.surface_properties_image()
        self.resistances_image()
        self.additional_options_point()
        # Model Selection tabs
        tabs = widgets.Tab(
            children=[
                self.w_model,
                self.site_page,
                self.met_page,
                self.spec_page,
                self.veg_page,
                self.res_page,
                self.opt_page])
        tabs.set_title(0, 'TSEB Model')
        tabs.set_title(1, 'Site Description')
        tabs.set_title(2, 'Meteorology')
        tabs.set_title(3, 'Spectral Properties')
        tabs.set_title(4, 'Canopy Description')
        tabs.set_title(5, 'Resistance Model')
        tabs.set_title(6, 'Additional Options')
        # Display widgets
        display(self.w_loadconfig)
        display(widgets.VBox([widgets.HTML('Select Radiometric Temperature Image(s)'),
                              widgets.HBox([self.w_T_R1_But, self.w_T_R1]),
                              widgets.HBox([self.w_T_R0_But, self.w_T_R0]),
                              widgets.HTML('Select View Zenith Angle data or type a constant value'),
                              widgets.HBox([self.w_VZA, self.w_VZAtxt]),
                              widgets.HTML('Select LAI data or type a constant value'),
                              widgets.HBox([self.w_LAI, self.w_LAItxt]),
                              widgets.HTML('Select Canopy Height data or type a constant value'),
                              widgets.HBox([self.w_Hc, self.w_Hctxt]),
                              widgets.HTML('Select Fractional Cover data or type a constant value'),
                              widgets.HBox([self.w_f_c, self.w_f_ctxt]),
                              widgets.HTML('Select Canopy width/height ratio or type a constant value'),
                              widgets.HBox([self.w_w_C, self.w_w_Ctxt]),
                              widgets.HTML('Select Green Fraction data or type a constant value'),
                              widgets.HBox([self.w_f_g, self.w_f_gtxt]),
                              widgets.HTML('Select Image Mask or set 0 to process the whole image'),
                              widgets.HBox([self.w_mask, self.w_masktxt])], background_color='#EEE'))
        display(widgets.HBox([self.w_output, self.w_outputtxt]))
        display(tabs)
        display(self.w_saveconfig)
        display(self.w_runmodel)
        # Handle interactions
        self.w_T_R1_But.on_click(
            lambda b: self._on_input_clicked(b, 'Surface Radiometric Temperature', self.w_T_R1))
        self.w_T_R0_But.on_click(
            lambda b: self._on_input_clicked(b, 'Morning Surface Radiometric Temperature', self.w_T_R0))
        self.w_VZA.on_click(
            lambda b: self._on_input_clicked(b, 'View Zenith Angle', self.w_VZAtxt))
        self.w_LAI.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf Area Index', self.w_LAItxt))
        self.w_f_c.on_click(
            lambda b: self._on_input_clicked(b, 'Fractional Cover', self.w_f_ctxt))
        self.w_f_g.on_click(
            lambda b: self._on_input_clicked(b, 'Green Fraction', self.w_f_gtxt))
        self.w_Hc.on_click(
            lambda b: self._on_input_clicked(b, 'Canopy Height', self.w_Hctxt))
        self.w_w_C.on_click(
            lambda b: self._on_input_clicked(b, 'Canopy Width/Height Ratio', self.w_w_Ctxt))
        self.w_mask.on_click(
            lambda b: self._on_input_clicked(b, 'Mask', self.w_masktxt))
        self.w_output.on_click(self._on_output_clicked)
        self.w_model.on_trait_change(self._on_model_change, 'value')
        self.w_loadconfig.on_click(self._on_loadconfig_clicked)
        self.w_saveconfig.on_click(self._on_saveconfig_clicked)
        self.w_res.on_trait_change(self._on_res_change, 'value')
        self.w_row.on_trait_change(self._on_row_change, 'value')
        self.w_G_form.on_trait_change(self._on_G_change, 'value')
        self.w_runmodel.on_click(self._on_runmodel_clicked)

        self.is_image = True

    def select_model(self):
        ''' Widget to select the TSEB model'''

        self.w_model = widgets.ToggleButtons(
            description='Select TSEB model to run:',
            options={
                'Priestley Taylor': 'TSEB_PT',
                'Dual-Time Difference': 'DTD',
                'Component Temperatures': 'TSEB_2T'},
            value=self.model)

    def define_site_description_time_series(self):
        '''Widgets for site description parameters'''

        self.w_lat = widgets.BoundedFloatText(
            value=self.lat, min=-90, max=90, description='Lat.', width=100)
        self.w_lon = widgets.BoundedFloatText(
            value=self.lon, min=-180, max=180, description='Lon.', width=100)
        self.w_alt = widgets.FloatText(
            value=self.alt, description='Alt.', width=100)
        self.w_stdlon = widgets.BoundedFloatText(
            value=self.stdlon, min=-180, max=180, description='Std. Lon.', width=100)
        self.w_z_u = widgets.BoundedFloatText(
            value=self.zu,
            min=0.001,
            description='Wind meas. height',
            width=100)
        self.w_z_T = widgets.BoundedFloatText(
            value=self.zt, min=0.001, description='T meas. height', width=100)
        self.site_page = widgets.VBox([widgets.HBox([self.w_lat,
                                                    self.w_lon,
                                                    self.w_alt,
                                                    self.w_stdlon]),
                                      widgets.HBox([self.w_z_u,
                                                    self.w_z_T])],
                                      background_color='#EEE')

    def define_site_description_image(self):
        '''Widgets for site description parameters'''

        self.w_latBut = widgets.Button(description='Browse Latitude Image')
        self.w_lat = widgets.Text(
            description='(Decimal degrees)', value='0', width=500)
        self.w_lonBut = widgets.Button(description='Browse Longitude Image')
        self.w_lon = widgets.Text(
            description='(Decimal degrees):', value='0', width=500)
        self.w_altBut = widgets.Button(description='Browse Altitude Image')
        self.w_alt = widgets.Text(
            description='(m):', value='0', width=500)
        self.w_stdlon_But = widgets.Button(description='Browse Standard Longitude Image')
        self.w_stdlon = widgets.Text(
            description='(Decimal degrees):', value='0', width=500)
        self.w_z_u_But = widgets.Button(description='Wind meas. height')
        self.w_z_u = widgets.Text(
            description='(m):', value=str(self.zu), width=500)
        self.w_z_T_But = widgets.Button(description='Air temp. meas. height')
        self.w_z_T = widgets.Text(
            description='(m):', value=str(self.zt), width=500)

        self.site_page = widgets.VBox([widgets.HTML('Select latitude image or type a constant value'),
                                      widgets.HBox([self.w_latBut, self.w_lat]),
                                      widgets.HTML('Select longitude image or type a constant value'),
                                      widgets.HBox([self.w_lonBut, self.w_lon]),
                                      widgets.HTML('Select altitude image or type a constant value'),
                                      widgets.HBox([self.w_altBut, self.w_alt]),
                                      widgets.HTML('Select standard longitude image or type a constant value'),
                                      widgets.HBox([self.w_stdlon_But, self.w_stdlon]),
                                      widgets.HTML('Select wind measurement height image or type a constant value'),
                                      widgets.HBox([self.w_z_u_But, self.w_z_u]),
                                      widgets.HTML('Select air temperature measurement height image or type a constant value'),
                                      widgets.HBox([self.w_z_T_But, self.w_z_T])])

        self.w_latBut.on_click(
            lambda b: self._on_input_clicked(b, 'Latitude', self.w_lat))
        self.w_lonBut.on_click(
            lambda b: self._on_input_clicked(b, 'Longitude', self.w_lon))
        self.w_altBut.on_click(
            lambda b: self._on_input_clicked(b, 'Altitude', self.w_alt))
        self.w_stdlon_But.on_click(
            lambda b: self._on_input_clicked(b, 'Standard Longitude', self.w_stdlon))
        self.w_z_u_But.on_click(
            lambda b: self._on_input_clicked(b, 'Wind Measurement Height', self.w_z_u))
        self.w_z_T_But.on_click(
            lambda b: self._on_input_clicked(b, 'Air Temperature Measurement Height', self.w_z_T))

    def spectral_properties_image(self):
        '''Widgets for site spectral properties'''

        self.w_rho_vis_C_But = widgets.Button(description='Leaf refl. PAR')
        self.w_rho_vis_C = widgets.Text(
            description=' ', value=str(self.rho_vis_C), width=500)
        self.w_tau_vis_C_But = widgets.Button(description='Leaf trans. PAR')
        self.w_tau_vis_C = widgets.Text(
            description=' ', value=str(self.tau_vis_C), width=500)
        self.w_rho_nir_C_But = widgets.Button(description='Leaf refl. NIR')
        self.w_rho_nir_C = widgets.Text(
            description=' ', value=str(self.rho_nir_C), width=500)
        self.w_tau_nir_C_But = widgets.Button(description='Leaf trans. NIR')
        self.w_tau_nir_C = widgets.Text(
            description=' ', value=str(self.tau_nir_C), width=500)

        self.w_rho_vis_S_But = widgets.Button(description='Soil refl. PAR')
        self.w_rho_vis_S = widgets.Text(
            description=' ', value=str(self.rho_vis_S), width=500)
        self.w_rho_nir_S_But = widgets.Button(description='Soil refl. NIR')
        self.w_rho_nir_S = widgets.Text(
            description=' ', value=str(self.rho_nir_S), width=500)
        self.w_emis_C_But = widgets.Button(description='Leaf emissivity')
        self.w_emis_C = widgets.Text(
            description=' ', value=str(self.emis_C), width=500)
        self.w_emis_S_But = widgets.Button(description='Soil emissivity')
        self.w_emis_S = widgets.Text(
            description=' ', value=str(self.emis_S), width=500)

        self.spec_page = widgets.VBox([widgets.HTML('Select leaf PAR reflectance image or type a constant value'),
                                      widgets.HBox([self.w_rho_vis_C_But, self.w_rho_vis_C]),
                                      widgets.HTML('Select leaf PAR transmitance image or type a constant value'),
                                      widgets.HBox([self.w_tau_vis_C_But, self.w_tau_vis_C]),
                                      widgets.HTML('Select leaf NIR reflectance image or type a constant value'),
                                      widgets.HBox([self.w_rho_nir_C_But, self.w_rho_nir_C]),
                                      widgets.HTML('Select leaf NIR transmitance image or type a constant value'),
                                      widgets.HBox([self.w_tau_nir_C_But, self.w_tau_nir_C]),
                                      widgets.HTML('Select soil PAR reflectance or image type a constant value'),
                                      widgets.HBox([self.w_rho_vis_S_But, self.w_rho_vis_S]),
                                      widgets.HTML('Select soil NIR reflectance or image type a constant value'),
                                      widgets.HBox([self.w_rho_nir_S_But, self.w_rho_nir_S]),
                                      widgets.HTML('Select leaf emissivity image or type a constant value'),
                                      widgets.HBox([self.w_emis_C_But, self.w_emis_C]),
                                      widgets.HTML('Select soil emissivity image or type a constant value'),
                                      widgets.HBox([self.w_emis_S_But, self.w_emis_S])])

        self.w_rho_vis_C_But.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf PAR Reflectance', self.w_rho_vis_C))
        self.w_tau_vis_C_But.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf PAR Transmitance', self.w_tau_vis_C))
        self.w_rho_nir_C_But.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf NIR Reflectance', self.w_rho_nir_C))
        self.w_tau_nir_C_But.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf NIR Transmitance', self.w_tau_nir_C))
        self.w_rho_vis_S_But.on_click(
            lambda b: self._on_input_clicked(b, 'Soil PAR Reflectance', self.w_rho_vis_S))
        self.w_rho_nir_S_But.on_click(
            lambda b: self._on_input_clicked(b, 'Soil NIR Reflectance', self.w_rho_nir_S))
        self.w_emis_C_But.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf Emissivity', self.w_emis_C))
        self.w_emis_S_But.on_click(
            lambda b: self._on_input_clicked(b, 'Soil Emissivity', self.w_emis_S))

    def spectral_properties_time_series(self):
        '''Widgets for site spectral properties'''

        self.w_rho_vis_C = widgets.BoundedFloatText(
            value=self.rho_vis_C, min=0, max=1, description='Leaf refl. PAR', width=80)
        self.w_tau_vis_C = widgets.BoundedFloatText(
            value=self.tau_vis_C, min=0, max=1, description='Leaf trans. PAR', width=80)
        self.w_rho_nir_C = widgets.BoundedFloatText(
            value=self.rho_nir_C, min=0, max=1, description='Leaf refl. NIR', width=80)
        self.w_tau_nir_C = widgets.BoundedFloatText(
            value=self.tau_nir_C, min=0, max=1, description='Leaf trans. NIR', width=80)

        self.w_rho_vis_S = widgets.BoundedFloatText(
            value=self.rho_vis_S, min=0, max=1, description='Soil refl. PAR', width=80)
        self.w_rho_nir_S = widgets.BoundedFloatText(
            value=self.rho_nir_S, min=0, max=1, description='Soil refl. NIR', width=80)
        self.w_emis_C = widgets.BoundedFloatText(
            value=self.emis_C, min=0, max=1, description='Leaf emissivity', width=80)
        self.w_emis_S = widgets.BoundedFloatText(
            value=self.emis_S, min=0, max=1, description='Soil emissivity', width=80)
        self.spec_page = widgets.VBox([widgets.HBox([self.w_rho_vis_C, self.w_tau_vis_C, self.w_rho_nir_C, self.w_tau_nir_C]), widgets.HBox(
            [self.w_rho_vis_S, self.w_rho_nir_S, self.w_emis_C, self.w_emis_S])], background_color='#EEE')

    def meteorology(self):
        '''Widgets for meteorological forcing'''
        self.w_DOY_But = widgets.Button(
            description='Browse Date of Year Image')
        self.w_DOY = widgets.Text(description=' ', value='1', width=500)
        self.w_time_But = widgets.Button(
            description='Browse Dec Time Image')
        self.w_time = widgets.Text(description='(h):', value='12', width=500)
        self.w_T_A1_But = widgets.Button(
            description='Browse Air Temperature Image')
        self.w_T_A1 = widgets.Text(description='(K):', value='0', width=500)
        self.w_T_A0_But = widgets.Button(
            description='Browse Sunrise Air Temperature Image')
        self.w_T_A0 = widgets.Text(description='(K):', value='0', width=500)
        self.w_T_A0_But.visible = False
        self.w_T_A0.visible = False
        self.w_S_dnBut = widgets.Button(
            description='Browse Shortwave Irradiance Image')
        self.w_S_dn = widgets.Text(description='(W/m2):', value='0', width=500)
        self.w_uBut = widgets.Button(description='Browse Wind Speed Image')
        self.w_u = widgets.Text(description='(m/s):', value='0.01', width=500)
        self.w_eaBut = widgets.Button(
            description='Browse Vapour Pressure Image')
        self.w_ea = widgets.Text(description='(mb):', value='0', width=500)

        met_text = widgets.HTML(
            'OPTIONAL: Leave empy to use estimated values for the following cells')
        self.w_L_dnBut = widgets.Button(
            description='Browse Longwave Irradiance Image')
        self.w_L_dn = widgets.Text(description='(W/m2):', value='', width=500)
        self.w_pBut = widgets.Button(description='Browse Pressure Image')
        self.w_p = widgets.Text(description='(mb):', value='', width=500)

        self.met_page = widgets.VBox([widgets.HTML('Select Day of Year image or type a constant value'),
                                     widgets.HBox([self.w_DOY_But, self.w_DOY]),
                                     widgets.HTML('Select time (decimal) image or type a constant value'),
                                     widgets.HBox([self.w_time_But, self.w_time]),
                                     widgets.HTML('Select air temperature image(s) or type a constant value'),
                                     widgets.HBox([self.w_T_A1_But, self.w_T_A1]),
                                     widgets.HBox([self.w_T_A0_But, self.w_T_A0]),
                                     widgets.HTML('Select shortwave irradiance image or type a constant value'),
                                     widgets.HBox([self.w_S_dnBut, self.w_S_dn]),
                                     widgets.HTML('Select wind speed image or type a constant value'),
                                     widgets.HBox([self.w_uBut, self.w_u]),
                                     widgets.HTML('Select vapour pressure image or type a constant value'),
                                     widgets.HBox([self.w_eaBut, self.w_ea]),
                                     widgets.HTML('<br>'),
                                     met_text,
                                     widgets.HTML('Select longwave irradiance image or type a constant value'),
                                     widgets.HBox([self.w_L_dnBut, self.w_L_dn]),
                                     widgets.HTML('Select pressure image or type a constant value'),
                                     widgets.HBox([self.w_pBut, self.w_p])], background_color='#EEE')

        self.w_DOY_But.on_click(
            lambda b: self._on_input_clicked(b, 'Day of Year', self.w_DOY))
        self.w_time_But.on_click(
            lambda b: self._on_input_clicked(b, 'Decimal Time', self.w_time))
        self.w_T_A0_But.on_click(
            lambda b: self._on_input_clicked(b, 'Sunrise Air Temperature', self.w_T_A0))
        self.w_T_A1_But.on_click(
            lambda b: self._on_input_clicked(b, 'Air Temperature', self.w_T_A1))
        self.w_S_dnBut.on_click(
            lambda b: self._on_input_clicked(b, 'Shortwave Irradiance', self.w_S_dn))
        self.w_uBut.on_click(
            lambda b: self._on_input_clicked(b, 'Wind Speed', self.w_u))
        self.w_eaBut.on_click(
            lambda b: self._on_input_clicked(b, 'Vapour Pressure', self.w_ea))
        self.w_L_dnBut.on_click(
            lambda b: self._on_input_clicked(b, 'Longwave Irradiance', self.w_L_dn))
        self.w_pBut.on_click(
            lambda b: self._on_input_clicked(b, 'Pressure', self.w_p))

    def surface_properties_time_series(self):
        '''Widgets for canopy properties'''

        self.w_PT = widgets.BoundedFloatText(
            value=self.max_PT, min=0, description="Max. alphaPT", width=80)
        self.w_LAD = widgets.BoundedFloatText(
            value=self.x_LAD, min=0, description="LIDF param.", width=80)
        self.w_LAD.visible = False
        self.w_leafwidth = widgets.BoundedFloatText(
            value=self.leaf_width, min=0.001, description="Leaf width", width=80)
        self.w_zsoil = widgets.BoundedFloatText(
            value=self.z0soil, min=0, description="soil roughness", width=80)
        # Landcover classes and values come from IGBP Land Cover Type Classification
        self.w_lc = widgets.Dropdown(
            options={
                    'WATER': 0,
                    'CONIFER EVERGREEN': 1,
                    'BROADLEAVED EVERGREEN': 2,
                    'CONIFER DECIDUOUS': 3,
                    'BROADLEAVED DECIDUOUS': 4,
                    'FOREST MIXED': 5,
                    'SHRUB CLOSED': 6,
                    'SHRUB OPEN': 7,
                    'SAVANNA WOODY': 8,
                    'SAVANNA': 9,
                    'GRASS': 10,
                    'WETLAND': 11,
                    'CROP': 12,
                    'URBAN': 13,
                    'CROP MOSAIC': 14,
                    'SNOW': 15,
                    'BARREN': 16
                    },
            value=self.landcover,
            description="Land Cover Type",
            width=200)
        lcText = widgets.HTML(value='''Land cover information is used to estimate roughness. <BR>
                                    For shrubs, conifers and broadleaves we use the model of <BR>
                                    Schaudt & Dickinson (2000) Agricultural and Forest Meteorology. <BR>
                                    For crops and grasses we use a fixed ratio of canopy heigh''', width=100)

        self.calc_row_options()
        self.veg_page = widgets.VBox([widgets.HBox([self.w_PT, self.w_LAD, self.w_leafwidth]),
                                      widgets.HBox([self.w_zsoil, self.w_lc, lcText]),
                                      widgets.HBox([self.w_row, self.w_rowaz])],
                                     background_color='#EEE')

    def surface_properties_image(self):
        '''Widgets for canopy properties'''

        self.w_PT_But = widgets.Button(
            description='Browse Initial alphaPT Image')
        self.w_PT = widgets.Text(description=' ', value=str(self.max_PT), width=500)
        self.w_LAD_But = widgets.Button(
            description='Browse Leaf Angle Distribution Image')
        self.w_LAD = widgets.Text(description='(degrees)', value=str(self.x_LAD), width=500)
        self.w_leafwidth_But = widgets.Button(
            description='Browse Leaf Width Image')
        self.w_leafwidth = widgets.Text(description='(m)', value=str(self.leaf_width), width=500)
        self.w_zsoil_But = widgets.Button(
            description='Browse Soil Roughness Image')
        self.w_zsoil = widgets.Text(description='(m)', value=str(self.z0soil), width=500)
        self.w_lc_But = widgets.Button(
            description='Browse Land Cover Image')
        # Landcover classes and values come from IGBP Land Cover Type Classification
        self.w_lc = widgets.Dropdown(
            options={
                'CROP': 12,
                'GRASS': 10,
                'SHRUB': 6,
                'CONIFER': 1,
                'BROADLEAVED': 4},
            value=self.landcover,
            description=" ",
            width=300)
        lcText = widgets.HTML(value='''Land cover information is used to estimate roughness. <BR>
                                    For shrubs, conifers and broadleaves we use the model of <BR>
                                    Schaudt & Dickinson (2000) Agricultural and Forest Meteorology. <BR>
                                    For crops and grasses we use a fixed ratio of canopy height<BR>''', width=600)

        self.calc_row_options()
        self.veg_page = widgets.VBox([widgets.HTML('Select alphaPT image or type a constant value'),
                                      widgets.HBox([self.w_PT_But, self.w_PT]),
                                      widgets.HTML('Select leaf angle distribution image or type a constant value'),
                                      widgets.HBox([self.w_LAD_But, self.w_LAD]),
                                      widgets.HTML('Select leaf width image or type a constant value'),
                                      widgets.HBox([self.w_leafwidth_But, self.w_leafwidth]),
                                      widgets.HTML('Select soil roughness image or type a constant value'),
                                      widgets.HBox([self.w_zsoil_But, self.w_zsoil]),
                                      widgets.HTML('Select landcover image or type a constant value'),
                                      widgets.HBox([self.w_lc_But, self.w_lc]),
                                      lcText,
                                      widgets.HBox([self.w_row, self.w_rowaz])], background_color='#EEE')

        self.w_PT_But.on_click(
            lambda b: self._on_input_clicked(b, 'Initial alphaPT', self.w_PT))
        self.w_LAD_But.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf Angle Distribution', self.w_LAD))
        self.w_leafwidth_But.on_click(
            lambda b: self._on_input_clicked(b, 'Leaf Width', self.w_leafwidth))
        self.w_zsoil_But.on_click(
            lambda b: self._on_input_clicked(b, 'Soil Roughness', self.w_zsoil))
        self.w_lc_But.on_click(
            lambda b: self._input_dropdown_clicked(b, 'Land Cover', self.w_lc))

    def calc_row_options(self):
        '''Widgets for canopy in rows'''

        self.w_row = widgets.Checkbox(
            description='Canopy in rows?', value=self.row)
        self.w_rowaz = widgets.BoundedFloatText(
            value=self.row_az,
            min=0,
            max=360,
            description='Row orientation',
            width=80)
        self.w_rowaz.visible = False

    def resistances_time_series(self):
        '''Widgets for resistance model selection'''

        self.w_res = widgets.ToggleButtons(
            description='Select TSEB model to run:',
            options={
                'Kustas & Norman 1999': 0,
                'Choudhury & Monteith 1988': 1,
                'McNaughton & Van der Hurk': 2},
            value=self.res,
            width=300)
        self.w_KN_b = widgets.BoundedFloatText(
            value=self.KN_b, min=0, description='KN99 b', width=80)
        self.w_KN_c = widgets.BoundedFloatText(
            value=self.KN_c, min=0, description='KN99 c', width=80)
        self.w_KN_C_dash = widgets.BoundedFloatText(
            value=self.KN_C_dash, min=0, max=9999, description="KN99 C'", width=80)
        self.KN_params_box = widgets.HBox([self.w_KN_b, self.w_KN_c, self.w_KN_C_dash])
        self.res_page = widgets.VBox([self.w_res, self.KN_params_box], background_color='#EEE')

    def resistances_image(self):
        '''Widgets for resistance model selection'''

        self.w_res = widgets.ToggleButtons(
            description='Select TSEB model to run:',
            options={
                'Kustas & Norman 1999': 0,
                'Choudhury & Monteith 1988': 1,
                'McNaughton & Van der Hurk': 2},
            value=self.res,
            width=300)

        self.w_PT_But = widgets.Button(
            description='Browse Initial alphaPT Image')
        self.w_PT = widgets.Text(description=' ', value=str(self.max_PT), width=500)

        self.w_KN_b_But = widgets.Button(description='Browse Resistance Parameter b Image')
        self.w_KN_b = widgets.Text(
            value=str(self.KN_b), description=' ', width=500)
        self.w_KN_c_But = widgets.Button(description=('Browse Resistance Parameter c image'))
        self.w_KN_c = widgets.Text(
            value=str(self.KN_c), description='(m s-1 K-1/3)', width=500)
        self.w_KN_C_dash_But = widgets.Button(description=("Browse Resistance Parameter C' Image"))
        self.w_KN_C_dash = widgets.Text(
            value=str(self.KN_C_dash), description="s1/2 m-1", width=500)
        self.KN_params_box = widgets.VBox([widgets.HTML('Select resistance parameter b image or type a constant value'),
                                           widgets.HBox([self.w_KN_b_But, self.w_KN_b]),
                                           widgets.HTML('Select resistance parameter c image or type a constant value'),
                                           widgets.HBox([self.w_KN_c_But, self.w_KN_c]),
                                           widgets.HTML('Select resistance parameter C\' image or type a constant value'),
                                           widgets.HBox([self.w_KN_C_dash_But, self.w_KN_C_dash])], background_color='#EEE')
        self.res_page = widgets.VBox([self.w_res, self.KN_params_box], background_color='#EEE')

        self.w_KN_b_But.on_click(
            lambda b: self._on_input_clicked(b, 'Resistance Parameter b', self.w_KN_b))
        self.w_KN_c_But.on_click(
            lambda b: self._on_input_clicked(b, 'Resistance Parameter c', self.w_KN_c))
        self.w_KN_C_dash_But.on_click(
            lambda b: self._on_input_clicked(b, 'Resistance Parameter C\'', self.w_KN_C_dash))

    def additional_options_point(self):
        '''Widgets for additional TSEB options'''

        self.calc_G_options()
        self.opt_page = widgets.VBox([
            self.w_G_form,
            self.w_Gratio,
            self.w_Gconstanttext,
            self.w_Gconstant,
            widgets.HBox([self.w_G_amp, self.w_G_phase, self.w_G_shape])], background_color='#EEE')

    def calc_G_options(self):
        '''Widgets for method for computing soil heat flux'''

        self.w_G_form = widgets.ToggleButtons(
            description='Select method for soil heat flux',
            options={
                'Ratio of soil net radiation': 1,
                'Constant or measured value': 0,
                'Time dependent (Santanelo & Friedl)': 2},
            value=self.G_form,
            width=300)
        self.w_Gratio = widgets.BoundedFloatText(
            value=self.Gratio, min=0, max=1, description='G ratio (G/Rn)', width=80)
        self.w_Gconstant = widgets.FloatText(
            value=self.Gconstant, description='Value (W m-2)', width=80)
        self.w_Gconstant.visible = False
        self.w_Gconstanttext = widgets.HTML(
            value="Set G value (W m-2), ignored if G is present in the input file")
        self.w_Gconstanttext.visible = False
        self.w_Gconstant.visible = False
        self.w_G_amp = widgets.BoundedFloatText(
            value=self.G_amp, min=0, max=1, description='Amplitude (G/Rn)', width=80)
        self.w_G_amp.visible = False
        self.w_G_phase = widgets.BoundedFloatText(
            value=self.G_phase, min=-24, max=24, description='Time Phase (h)', width=80)
        self.w_G_phase.visible = False
        self.w_G_shape = widgets.BoundedFloatText(
            value=self.G_shape, min=0, max=24, description='Time shape (h)', width=80)
        self.w_G_shape.visible = False

    def get_data_TSEB_widgets(self, is_image):
        '''Parses the parameters in the GUI to TSEB variables for running TSEB'''

        self.params['model'] = self.w_model.value

        self.params['lat'] = self.w_lat.value
        self.params['lon'] = self.w_lon.value
        self.params['alt'] = self.w_alt.value
        self.params['stdlon'] = self.w_stdlon.value
        self.params['z_u'] = self.w_z_u.value
        self.params['z_T'] = self.w_z_T.value

        self.params['emis_C'] = self.w_emis_C.value
        self.params['emis_S'] = self.w_emis_S.value
        self.params['rho_vis_C'] = self.w_rho_vis_C.value
        self.params['tau_vis_C'] = self.w_tau_vis_C.value
        self.params['rho_nir_C'] = self.w_rho_nir_C.value
        self.params['tau_nir_C'] = self.w_tau_nir_C.value
        self.params['rho_vis_S'] = self.w_rho_vis_S.value
        self.params['rho_nir_S'] = self.w_rho_nir_S.value

        self.params['alpha_PT'] = self.w_PT.value
        self.params['x_LAD'] = self.w_LAD.value
        self.params['leaf_width'] = self.w_leafwidth.value
        self.params['z0_soil'] = self.w_zsoil.value
        self.params['landcover'] = self.w_lc.value

        self.params['resistance_form'] = self.w_res.value
        self.params['KN_b'] = self.w_KN_b.value
        self.params['KN_c'] = self.w_KN_c.value
        self.params['KN_C_dash'] = self.w_KN_C_dash.value

        if self.w_row.value == 0:
            self.params['calc_row'] = [0, 0]
        else:
            self.params['calc_row'] = [1, self.w_rowaz.value]

        if self.w_G_form.value == 0:
            self.params['G_form'] = [[0], self.w_Gconstant.value]
        elif self.w_G_form.value == 1:
            self.params['G_form'] = [[1], self.w_Gratio.value]
        elif self.w_G_form.value == 2:
            self.params['G_form'] = [[2, self.w_G_amp.value,
                                     self.w_G_phase.value, self.w_G_shape.value], 12.0]

        self.params['output_file'] = self.w_outputtxt.value
        self.outputFile = self.params['output_file']

        if is_image:
            # Get all the input parameters
            self.params['T_R1'] = self.w_T_R1.value
            self.params['VZA'] = self.w_VZAtxt.value
            self.params['LAI'] = self.w_LAItxt.value
            self.params['h_C'] = self.w_Hctxt.value
            self.params['f_c'] = self.w_f_ctxt.value
            self.params['f_g'] = self.w_f_gtxt.value
            self.params['w_C'] = self.w_w_Ctxt.value
            self.params['input_mask'] = self.w_masktxt.value

            self.params['DOY'] = self.w_DOY.value
            self.params['time'] = self.w_time.value
            self.params['T_A1'] = self.w_T_A1.value
            self.params['S_dn'] = self.w_S_dn.value
            self.params['u'] = self.w_u.value
            self.params['ea'] = self.w_ea.value
            self.params['L_dn'] = self.w_L_dn.value
            self.params['p'] = self.w_p.value

            if self.params['model'] == 'DTD':
                self.params['T_R0'] = self.w_T_R0.value
                self.params['T_A0'] = self.w_T_A0.value
        else:
            self.params['input_file'] = self.w_inputtxt.value
            self.params['f_c'] = self.f_c
            self.params['f_g'] = self.f_g
            self.params['w_C'] = self.w_c

        self.params['water_stress'] = False

        self.ready = True

    def _on_model_change(self, name, value):
        '''Behaviour when TSEB model is changed'''

        if value == 'DTD':
            self.w_T_R0_But.visible = True
            self.w_T_R0.visible = True
            self.w_T_A0_But.visible = True
            self.w_T_A0.visible = True
        else:
            self.w_T_R0_But.visible = False
            self.w_T_R0.visible = False
            self.w_T_A0_But.visible = False
            self.w_T_A0.visible = False

    def _on_row_change(self, name, value):
        '''Behaviour when selecting a canopy in row'''

        if value == 0:
            self.w_rowaz.visible = False
        else:
            self.w_rowaz.visible = True

    def _on_res_change(self, name, value):
        '''Behaviour when changing the resistance model'''

        if value == 0:
            self.KN_params_box.visible = True
        else:
            self.KN_params_box.visible = False

    def _on_G_change(self, name, value):
        '''Behaviour when changing the soil heat flux model'''

        if value == 0:
            self.w_Gratio.visible = False
            self.w_Gconstant.visible = True
            self.w_Gconstanttext.visible = True
            self.w_G_amp.visible = False
            self.w_G_phase.visible = False
            self.w_G_shape.visible = False
        elif value == 1:
            self.w_Gratio.visible = True
            self.w_Gconstant.visible = False
            self.w_Gconstanttext.visible = False
            self.w_G_amp.visible = False
            self.w_G_phase.visible = False
            self.w_G_shape.visible = False
        elif value == 2:
            self.w_Gratio.visible = False
            self.w_Gconstant.visible = False
            self.w_Gconstanttext.visible = False
            self.w_G_amp.visible = True
            self.w_G_phase.visible = True
            self.w_G_shape.visible = True

    def _on_loadconfig_clicked(self, b):
        '''Reads a configuration file and parses its data into the GUI'''

        input_file = self._get_input_filename(
            title='Select Input Configuration File')
        if not input_file:
            return
        config_data = self.parse_input_config(input_file)
        self.get_data(config_data, is_image=self.is_image)

        # Update the widget fields
        self.w_model.value = self.params['model']
        self.w_lat.value = self.params['lat']
        self.w_lon.value = self.params['lon']
        self.w_alt.value = self.params['alt']
        self.w_stdlon.value = self.params['stdlon']
        self.w_z_u.value = self.params['z_u']
        self.w_z_T.value = self.params['z_T']
        self.w_emis_C.value = self.params['emis_C']
        self.w_emis_S.value = self.params['emis_S']
        self.w_rho_vis_C.value = self.params['rho_vis_C']
        self.w_tau_vis_C.value = self.params['tau_vis_C']
        self.w_rho_nir_C.value = self.params['rho_nir_C']
        self.w_tau_nir_C.value = self.params['tau_nir_C']
        self.w_rho_vis_S.value = self.params['rho_vis_S']
        self.w_rho_nir_S.value = self.params['rho_nir_S']
        self.w_PT.value = self.params['alpha_PT']
        self.w_LAD.value = self.params['x_LAD']
        self.w_leafwidth.value = self.params['leaf_width']
        self.w_zsoil.value = self.params['z0_soil']
        try:
            self.params['landcover'] = int(self.params['landcover'])
            if self.params['landcover'] not in self.w_lc.options.values():
                    options = self.w_lc.options.copy()
                    options.update({self.params['landcover']: self.params['landcover']})
                    self.w_lc.options = options
            self.w_lc.value = self.params['landcover']
        except ValueError:
            pass
        g_form = self.params['G_form'][0][0]
        self.w_G_form.value = self.params['G_form'][0][0]
        if g_form == 0:
            self.w_Gconstant.value = self.params['G_form'][1]
        if g_form == 1:
            self.w_Gratio.value = self.params['G_form'][1]
        if g_form == 2:
            self.w_G_amp.value = self.params['G_form'][0][1]
            self.w_G_phase.value = self.params['G_form'][0][2]
            self.w_G_shape.value = self.params['G_form'][0][3]
        self.w_outputtxt.value = self.params['output_file']
        self.w_res.value = int(self.params['resistance_form'])
        self.w_KN_b.value = self.params['KN_b']
        self.w_KN_c.value = self.params['KN_c']
        self.w_KN_C_dash.value = self.params['KN_C_dash']

        if self.is_image:
            self.w_T_R1.value = str(self.params['T_R1']).strip('"')
            self.w_VZAtxt.value = str(self.params['VZA']).strip('"')
            self.w_LAItxt.value = str(self.params['LAI']).strip('"')
            self.w_Hctxt.value = str(self.params['h_C']).strip('"')
            self.w_f_ctxt.value = str(self.params['f_c']).strip('"')
            self.w_f_gtxt.value = str(self.params['f_g']).strip('"')
            self.w_w_Ctxt.value = str(self.params['w_C']).strip('"')
            self.w_masktxt.value = str(self.params['input_mask']).strip('"')
            self.w_DOY.value = self.params['DOY']
            self.w_time.value = self.params['time']
            self.w_T_A1.value = self.params['T_A1']
            self.w_S_dn.value = self.params['S_dn']
            self.w_u.value = self.params['u']
            self.w_ea.value = self.params['ea']
            self.w_L_dn.value = str(self.params['L_dn']).strip('"')
            self.w_p.value = str(self.params['p']).strip('"')
            if self.params['model'] == "DTD":
                self.w_T_R0.value = str(self.params['T_R0']).strip('"')
                self.w_T_A0.value = self.params['T_A0']
        else:
            self.w_inputtxt.value = str(self.params['input_file']).strip('"')

    def _on_saveconfig_clicked(self, b):
        '''Opens a configuration file and writes the parameters in the GUI into the file'''

        output_file = self._get_output_filename(
            title='Select Output Configuration File')
        if not output_file:
            return
        try:
            fid = open(output_file, 'w')
        except IOError:
            print('Could not write ' + output_file)
            return
        fid.write('# Input files\n')
        if self.is_image:
            fid.write('T_R1=' + str(self.w_T_R1.value) + '\n')
            fid.write('T_R0=' + str(self.w_T_R0.value) + '\n')
            fid.write('VZA=' + str(self.w_VZAtxt.value) + '\n')
            fid.write('LAI=' + str(self.w_LAItxt.value) + '\n')
            fid.write('f_c=' + str(self.w_f_ctxt.value) + '\n')
            fid.write('h_C=' + str(self.w_Hctxt.value) + '\n')
            fid.write('f_g=' + str(self.w_f_gtxt.value) + '\n')
            fid.write('w_C=' + str(self.w_w_Ctxt.value) + '\n')
            fid.write('input_mask=' + str(self.w_masktxt.value) + '\n')

            fid.write('\n# Output file\n')
            fid.write('output_file=' + str(self.w_outputtxt.value) + '\n')

            fid.write('\n# Meteorological data\n')
            fid.write('DOY=' + str(self.w_DOY.value) + '\n')
            fid.write('time=' + str(self.w_time.value) + '\n')
            fid.write('T_A1=' + str(self.w_T_A1.value) + '\n')
            fid.write('S_dn=' + str(self.w_S_dn.value) + '\n')
            fid.write('u=' + str(self.w_u.value) + '\n')
            fid.write('ea=' + str(self.w_ea.value) + '\n')
            fid.write('p=' + str(self.w_p.value) + '\n')
            fid.write('L_dn=' + str(self.w_L_dn.value) + '\n')
            fid.write('T_A0=' + str(self.w_T_A0.value) + '\n')

        else:
            fid.write('input_file=' + str(self.w_inputtxt.value) + '\n')
            fid.write('\n# Output file\n')
            fid.write('output_file=' + str(self.w_outputtxt.value) + '\n')

        # Write the commom fields
        fid.write('\n# Model Selection\n')
        fid.write('model=' + str(self.w_model.value) + '\n')

        fid.write('\n# Site Description\n')
        fid.write('lat=' + str(self.w_lat.value) + '\n')
        fid.write('lon=' + str(self.w_lon.value) + '\n')
        fid.write('alt=' + str(self.w_alt.value) + '\n')
        fid.write('stdlon=' + str(self.w_stdlon.value) + '\n')
        fid.write('z_u=' + str(self.w_z_u.value) + '\n')
        fid.write('z_T=' + str(self.w_z_T.value) + '\n')

        fid.write('\n# Spectral Properties\n')
        fid.write('emis_C=' + str(self.w_emis_C.value) + '\n')
        fid.write('emis_S=' + str(self.w_emis_S.value) + '\n')
        fid.write('rho_vis_C=' + str(self.w_rho_vis_C.value) + '\n')
        fid.write('rho_nir_C=' + str(self.w_rho_nir_C.value) + '\n')
        fid.write('tau_vis_C=' + str(self.w_tau_vis_C.value) + '\n')
        fid.write('tau_nir_C=' + str(self.w_tau_nir_C.value) + '\n')
        fid.write('rho_vis_S=' + str(self.w_rho_vis_S.value) + '\n')
        fid.write('rho_nir_S=' + str(self.w_rho_nir_S.value) + '\n')

        fid.write('\n# Surface Properties\n')
        fid.write('alpha_PT=' + str(self.w_PT.value) + '\n')
        fid.write('x_LAD=' + str(self.w_LAD.value) + '\n')
        fid.write('leaf_width=' + str(self.w_leafwidth.value) + '\n')
        fid.write('z0_soil=' + str(self.w_zsoil.value) + '\n')
        fid.write('landcover=' + str(self.w_lc.value) + '\n')

        fid.write('\n# Additional Options\n')
        fid.write('resistance_form=' + str(self.w_res.value) + '\n')
        fid.write('KN_b=' + str(self.w_KN_b.value) + '\n')
        fid.write('KN_c=' + str(self.w_KN_c.value) + '\n')
        fid.write('KN_C_dash=' + str(self.w_KN_C_dash.value) + '\n')
        fid.write('calc_row=' + str(int(self.w_row.value)) + '\n')
        fid.write('row_az=' + str(self.w_rowaz.value) + '\n')
        fid.write('G_form=' + str(self.w_G_form.value) + '\n')
        fid.write('G_constant=' + str(self.w_Gconstant.value) + '\n')
        fid.write('G_ratio=' + str(self.w_Gratio.value) + '\n')
        fid.write('G_amp=' + str(self.w_G_amp.value) + '\n')
        fid.write('G_phase=' + str(self.w_G_phase.value) + '\n')
        fid.write('G_shape=' + str(self.w_G_shape.value) + '\n')
        fid.flush()
        fid.close()
        del fid
        print('Saved Configuration File')

    def _on_input_clicked(self, b, name, value_widget):
        value_widget.value = self._get_input_filename("Select "+name+" Image")

    def _input_dropdown_clicked(self, b, name, value_widget):
        filename = self._get_input_filename("Select "+name+" Image")
        options = value_widget.options.copy()
        options.update({filename: filename})
        value_widget.options = options
        value_widget.value = filename

    def _get_input_filename(self, title='Select Input File'):
        root, askopenfilename, _ = self._setup_tkinter()
        # show an "Open" dialog box and return the path to the selected file
        input_file = askopenfilename(parent=root, title=title)
        root.destroy()  # Destroy the GUI
        return input_file

    def _on_output_clicked(self, b):
        '''Behaviour when clicking the output file button'''
        self.w_outputtxt.value = self._get_output_filename()

    def _get_output_filename(self, title='Select Output File'):
        root, _, asksaveasfilename = self._setup_tkinter()
        # show an "Open" dialog box and return the path to the selected file
        output_file = asksaveasfilename(title=title)
        root.destroy()  # Destroy the GUI
        return output_file

    def _setup_tkinter(self):
        '''Creates a Tkinter input file dialog'''

        # Import Tkinter GUI widgets
        if sys.version_info.major == 2:
            from tkFileDialog import askopenfilename, asksaveasfilename
            import Tkinter as tk
        else:
            from tkinter.filedialog import askopenfilename, asksaveasfilename
            import tkinter as tk

        # Code below is to make sure the file dialog appears above the
        # terminal/browser
        # Based on
        # http://stackoverflow.com/questions/3375227/how-to-give-tkinter-file-dialog-focus

        # Make a top-level instance and hide since it is ugly and big.
        root = tk.Tk()
        root.withdraw()

        # Make it almost invisible - no decorations, 0 size, top left corner.
        root.overrideredirect(True)
        root.geometry('0x0+0+0')

        # Show window again and lift it to top so it can get focus,
        # otherwise dialogs will end up behind the terminal.
        root.deiconify()
        root.lift()
        root.focus_force()

        return root, askopenfilename, asksaveasfilename

    def _on_runmodel_clicked(self, b):
        # Change the colour of the button to know it is running
        self.w_runmodel.background_color = 'yellow'
        # Get the data from the widgets
        self.get_data_TSEB_widgets(is_image=self.is_image)
        # run TSEB
        self.run(is_image=self.is_image)
        # Change the colour of the button to know it has finished
        self.w_runmodel.background_color = 'green'
