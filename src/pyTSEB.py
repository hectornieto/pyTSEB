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

    import pyTSEB # Import pyTSEB module
    tseb=pyTSEB.PyTSEB() # Create a PyTSEB instance
    tseb.PointTimeSeriesWidget() # Launches the GUI

then to run pyTSEB.

.. code-block:: ipython

    tseb.GetDataTSEBWidgets(isImage=False)# Get the data from the widgets
    tseb.RunTSEBLocalImage()# Run TSEB

Similarly, to configure and run TSEB for an image.

.. code-block:: ipython

    import pyTSEB # Import pyTSEB module
    tseb=pyTSEB.PyTSEB() # Create a PyTSEB instance
    tseb.PointLocalImageWidget() # Launches the GUI
    tseb.GetDataTSEBWidgets(isImage=True)# Get the data from the widgets
    tseb.RunTSEBLocalImage()# Run TSEB

Parsing directly a configuration file
-------------------------------------
You can also parse direcly into TSEB a configuration file previouly created.

>>> tseb=PyTSEB()
>>> configData=tseb.parseInputConfig(configFile,isImage=True) #Read the data from the configuration file into a python dictionary
>>> tseb.GetDataTSEB(configData,isImage=True) # Parse the data from the dictionary to TSEB
>>> tseb.RunTSEBLocalImage() # Run TSEB

see the guidelines for input and configuration file preparation in :doc:`README_Notebooks`.

"""

class PyTSEB():
    
    def __init__(self):
        '''Initialize input variables  with default  values'''
        self.InputFile='./Input/ExampleInput.txt'
        self.OutputTxtFile='./Output/test.txt'
        self.OutputImageFile='./Output/test.tif'
        
        # MOdel to run
        self.model='TSEB_PT'
        # Site Description
        self.lat=36.95
        self.lon=2.33
        self.alt=200
        self.stdlon=15
        self.zu=2.5
        self.zt=3.5
        # Spectral Properties
        self.rhovis=0.07
        self.rhonir=0.32
        self.tauvis=0.08
        self.taunir=0.33
        self.rsoilv=0.15
        self.rsoiln=0.25
        self.emisVeg=0.98
        self.emisGrd=0.95
        # Surface Properties
        self.max_PT=1.26
        self.x_LAD=1.0
        self.leaf_width=0.1
        self.z0soil=0.01
        self.LANDCOVER=11
        # Soil Heat Flux calculation
        self.CalcG=1
        self.Gconstant=0
        self.Gratio=0.35
        self.GAmp=0.35
        self.Gphase=3
        self.Gshape=24
        # Default Vegetation variables
        self.f_c=1.0
        self.f_g=1.0
        self.wc=1.0
        # Output variables saved in images
        self.fields=('H1','LE1','R_n1','G1')
        # Ancillary output variables
        self.anc_fields=('H_C1','LE_C1','LE_partition','T_C1', 'T_S1','R_ns1','R_nl1', 'u_friction', 'L')
        # File Configuration variables
        self.input_commom_vars=('TSEB_MODEL','lat','lon','altitude','stdlon',
                   'z_t','z_u','emisVeg','emisGrd','rhovis','tauvis',
                   'rhonir','taunir','rsoilv','rsoiln','Max_alpha_PT',
                   'x_LAD','z0_soil','LANDCOVER','leaf_width',
                   'CalcG','G_constant','G_ratio','GAmp','Gphase','Gshape','OutputFile')
        self.input_image_vars=['Input_LST','Input_VZA','USE_MASK','Input_LAI','Input_Fc',
                         'Input_Fg','Input_Hc','Input_Wc','OutputFile','Time','DOY','Ta_1',
                         'Ta_0','u','ea', 'Sdn', 'Ldn','p']
        self.input_point_vars=['InputFile']
        
    def PointTimeSeriesWidget(self):
        '''Creates a jupyter notebook GUI for running TSEB for a point time series dataset'''
        import ipywidgets as widgets
        from IPython.display import display
        # Load and save configuration buttons
        self.w_loadconfig=widgets.Button(description='Load Configuration File')
        self.w_saveconfig=widgets.Button(description='Save Configuration File')
        # Input and output ascii files
        self.w_input=widgets.Button(description='Select Input File')
        self.w_inputtxt=widgets.Text(description='Input File :', value=self.InputFile, width=500)
        self.w_output=widgets.Button(description='Select Output File')
        self.w_outputtxt=widgets.Text(description='Output File :', value=self.OutputTxtFile,width=500)
        # Create TSEB options widgets
        self.SelectModel()
        self.DefineSiteDescription()
        self.SpectralProperties()
        self.SurfaceProperties()
        self.AdditionalOptionsPoint()
        
        # Model Selection tabs
        tabs = widgets.Tab(children=[self.w_model,self.sitePage,self.specPage,self.vegPage,self.optPage])
        tabs.set_title(0, 'TSEB Model')
        tabs.set_title(1, 'Site Description')
        tabs.set_title(2, 'Spectral Properties')
        tabs.set_title(3, 'Canopy Description')
        tabs.set_title(4, 'Additional Options')
        # Display widgets
        display(self.w_loadconfig)        
        display(widgets.HBox([self.w_input,self.w_inputtxt]))
        display(widgets.HBox([self.w_output,self.w_outputtxt]))
        display(tabs)
        display(self.w_saveconfig)
        # Handle interactions        
        self.w_CalcG.on_trait_change(self._on_G_change, 'value')
        self.w_input.on_click(self._on_input_clicked)    
        self.w_output.on_click(self._on_output_clicked)
        self.isImage=False
        self.w_loadconfig.on_click(self._on_loadconfig_clicked)
        self.w_saveconfig.on_click(self._on_saveconfig_clicked)
               
    def LocalImageWidget(self):
        '''Creates a jupyter notebook GUI for running TSEB for an image'''
        import ipywidgets as widgets
        from IPython.display import display
        # Load and save configuration buttons
        self.w_loadconfig=widgets.Button(description='Load Configuration File')
        self.w_saveconfig=widgets.Button(description='Save Configuration File')
        # Input and output images
        self.w_LST=widgets.Button(description='Browse Trad Image')
        self.w_LSTtxt=widgets.Text(description='LST:',value='Type Trad File', width=500)
        self.w_VZA=widgets.Button(description='Browse VZA Image')
        self.w_VZAtxt=widgets.Text(description='VZA:',value='0', width=500)
        self.w_LAI=widgets.Button(description='Browse LAI Image')
        self.w_LAItxt=widgets.Text(description='LAI:',value='1', width=500)
        self.w_Hc=widgets.Button(description='Browse H canopy Image')
        self.w_Hctxt=widgets.Text(description='H canopy:',value='1', width=500)
        self.w_Wc=widgets.Button(description='Browse Canopy widht/height Image')
        self.w_Wctxt=widgets.Text(description='Wc canopy:',value=str(self.wc), width=500)
        self.w_Fc=widgets.Button(description='Browse F cover Image')
        self.w_Fctxt=widgets.Text(description='F cover:',value=str(self.f_c), width=500)
        self.w_Fg=widgets.Button(description='Browse F green Image')
        self.w_Fgtxt=widgets.Text(description='F green:',value=str(self.f_g), width=500)
        self.w_mask=widgets.Button(description='Browse Image Mask')
        self.w_masktxt=widgets.Text(description='Mask:',value='0', width=500)
        self.w_output=widgets.Button(description='Select Output File')
        self.w_outputtxt=widgets.Text(description='Output File :', value=self.OutputImageFile, width=500)
        # Create TSEB options widgets
        self.SelectModel()
        self.DefineSiteDescription()
        self.Meteorology()
        self.SpectralProperties()
        self.SurfaceProperties()
        self.AdditionalOptionsPoint()
        # Model Selection tabs
        tabs = widgets.Tab(children=[self.w_model,self.sitePage,self.metPage,self.specPage,self.vegPage,self.optPage])
        tabs.set_title(0, 'TSEB Model')
        tabs.set_title(1, 'Site Description')
        tabs.set_title(2, 'Meteorology')
        tabs.set_title(3, 'Spectral Properties')
        tabs.set_title(4, 'Canopy Description')
        tabs.set_title(5, 'Additional Options')
        # Display widgets
        display(self.w_loadconfig)        
        display(widgets.VBox([widgets.HTML('Select Radiometric Temperature Image'),
            widgets.HBox([self.w_LST,self.w_LSTtxt]),
            widgets.HTML('Select View Zenith Angle data or type a constant value'),
            widgets.HBox([self.w_VZA,self.w_VZAtxt]),
            widgets.HTML('Select LAI data or type a constant value'),
            widgets.HBox([self.w_LAI,self.w_LAItxt]),
            widgets.HTML('Select Canopy Height data or type a constant value'),
            widgets.HBox([self.w_Hc,self.w_Hctxt]),
            widgets.HTML('Select Fractional Cover data or type a constant value'),
            widgets.HBox([self.w_Fc,self.w_Fctxt]),
            widgets.HTML('Select Canopy width/height ratio or type a constant value'),
            widgets.HBox([self.w_Wc,self.w_Wctxt]),
            widgets.HTML('Select Green Fraction data or type a constant value'),
            widgets.HBox([self.w_Fg,self.w_Fgtxt]),
            widgets.HTML('Select Image Mask or set 0 to process the whole image'),
            widgets.HBox([self.w_mask,self.w_masktxt])], background_color='#EEE'))
        display(widgets.HBox([self.w_output,self.w_outputtxt]))
        display(tabs)
        display(self.w_saveconfig) 
        # Handle interactions
        self.w_LST.on_click(self._on_inputLST_clicked)    
        self.w_VZA.on_click(self._on_inputVZA_clicked)    
        self.w_LAI.on_click(self._on_inputLAI_clicked)  
        self.w_Fc.on_click(self._on_inputFc_clicked)  
        self.w_Fg.on_click(self._on_inputFg_clicked)  
        self.w_Hc.on_click(self._on_inputHc_clicked)  
        self.w_Wc.on_click(self._on_inputWc_clicked)
        self.w_mask.on_click(self._on_inputmask_clicked)
        self.w_output.on_click(self._on_output_clicked)    
        self.w_model.on_trait_change(self._on_model_change,'value')
        self.w_CalcG.on_trait_change(self._on_G_change, 'value')
        self.w_loadconfig.on_click(self._on_loadconfig_clicked)
        self.w_saveconfig.on_click(self._on_saveconfig_clicked)

        self.isImage=True

    def _on_loadconfig_clicked(self,b):
        '''Reads a configuration file and parses its data into the GUI'''
        
        InputFile=self.GetInputFileName(title='Select Input Configuration File')
        if not InputFile:return
        configdata=self.parseInputConfig(InputFile,isImage=self.isImage)
        # Update the widget fields
        self.w_model.value=configdata['TSEB_MODEL']
        self.w_lat.value=configdata['lat']
        self.w_lon.value=configdata['lon']
        self.w_alt.value=configdata['altitude']
        self.w_stdlon.value=str(int(float(configdata['stdlon'])/15))
        self.w_zu.value=configdata['z_u']
        self.w_zt.value=configdata['z_t']
        self.w_emisVeg.value=configdata['emisVeg']
        self.w_emisSoil.value=configdata['emisGrd']
        self.w_rhovis.value=configdata['rhovis']
        self.w_tauvis.value=configdata['tauvis']
        self.w_rhonir.value=configdata['rhonir']
        self.w_taunir.value=configdata['taunir']
        self.w_rsoilv.value=configdata['rsoilv']
        self.w_rsoiln.value=configdata['rsoiln']
        self.w_PT.value=configdata['Max_alpha_PT']
        self.w_LAD.value=configdata['x_LAD']
        self.w_leafwidth.value=configdata['leaf_width']
        self.w_zsoil.value=configdata['z0_soil']
        self.w_lc.value=int(configdata['LANDCOVER'])
        self.w_CalcG.value=int(configdata['CalcG'])
        self.w_Gconstant.value=configdata['G_constant']
        self.w_Gratio.value=configdata['G_ratio']
        self.w_GAmp.value=configdata['GAmp']
        self.w_Gphase.value=configdata['Gphase']
        self.w_Gshape.value=configdata['Gshape']
        self.w_outputtxt.value=configdata['OutputFile']
        
        if self.isImage:
            self.w_LSTtxt.value=str(configdata['Input_LST']).strip('"')
            self.w_VZAtxt.value=str(configdata['Input_VZA']).strip('"')
            self.w_LAItxt.value=str(configdata['Input_LAI']).strip('"')
            self.w_Hctxt.value=str(configdata['Input_Hc']).strip('"')
            self.w_Fctxt.value=str(configdata['Input_Fc']).strip('"')
            self.w_Fgtxt.value=str(configdata['Input_Fg']).strip('"')
            self.w_Wctxt.value=str(configdata['Input_Wc']).strip('"')
            self.w_masktxt.value=str(configdata['USE_MASK']).strip('"')
            self.w_DOY.value=configdata['DOY']
            self.w_Time.value=configdata['Time']
            self.w_Ta_1.value=configdata['Ta_1']
            self.w_Sdn.value=configdata['Sdn']
            self.w_u.value=configdata['u']
            self.w_ea.value=configdata['ea']
            self.w_Ldn.value=str(configdata['Ldn']).strip('"')
            self.w_p.value=str(configdata['p']).strip('"')
            self.w_Ta_0.value=configdata['Ta_0']
        else:
            self.w_inputtxt.value=str(configdata['InputFile']).strip('"')

    def _on_saveconfig_clicked(self,b):
        '''Opens a configuration file and writes the parameters in the GUI into the file'''
        OutputFile=self.GetOutputFileName(title='Select Output Configuration File')
        if not OutputFile: return
        try:
            fid=open(OutputFile,'w')
        except IOError:
            print('Could not write ' +OutputFile)
            return
        fid.write('# Input files\n')
        if self.isImage:
            fid.write('Input_LST='+str(self.w_LSTtxt.value)+'\n')
            fid.write('Input_VZA='+str(self.w_VZAtxt.value)+'\n')
            fid.write('Input_LAI='+str(self.w_LAItxt.value)+'\n')
            fid.write('Input_Fc='+str(self.w_Fctxt.value)+'\n')
            fid.write('Input_Hc='+str(self.w_Hctxt.value)+'\n')
            fid.write('Input_Fg='+str(self.w_Fgtxt.value)+'\n')
            fid.write('Input_Wc='+str(self.w_Wctxt.value)+'\n')
            fid.write('USE_MASK='+str(self.w_masktxt.value)+'\n')

            fid.write('\n# Output file\n')
            fid.write('OutputFile='+str(self.w_outputtxt.value)+'\n')
            
            fid.write('\n# Meteorological data\n')
            fid.write('DOY='+str(self.w_DOY.value)+'\n')
            fid.write('Time='+str(self.w_Time.value)+'\n')
            fid.write('Ta_1='+str(self.w_Ta_1.value)+'\n')
            fid.write('Sdn='+str(self.w_Sdn.value)+'\n')
            fid.write('u='+str(self.w_u.value)+'\n')
            fid.write('ea='+str(self.w_ea.value)+'\n')
            fid.write('p='+str(self.w_p.value)+'\n')
            fid.write('Ldn='+str(self.w_Ldn.value)+'\n')
            fid.write('Ta_0='+str(self.w_Ta_0.value)+'\n')

        else:
            fid.write('InputFile='+str(self.w_inputtxt.value)+'\n')
            fid.write('\n# Output file\n')
            fid.write('OutputFile='+str(self.w_outputtxt.value)+'\n')
 
        # Write the commom fields
        fid.write('\n# Model Selection\n')
        fid.write('TSEB_MODEL='+str(self.w_model.value)+'\n')
        
        fid.write('\n# Site Description\n')
        fid.write('lat='+str(self.w_lat.value)+'\n')
        fid.write('lon='+str(self.w_lon.value)+'\n')
        fid.write('altitude='+str(self.w_alt.value)+'\n')
        fid.write('stdlon='+str(float(self.w_stdlon.value)*15)+'\n')
        fid.write('z_u='+str(self.w_zu.value)+'\n')          
        fid.write('z_t='+str(self.w_zt.value)+'\n')

        fid.write('\n# Spectral Properties\n')
        fid.write('emisVeg='+str(self.w_emisVeg.value)+'\n')
        fid.write('emisGrd='+str(self.w_emisSoil.value)+'\n')
        fid.write('rhovis='+str(self.w_rhovis.value)+'\n')
        fid.write('rhonir='+str(self.w_rhonir.value)+'\n')
        fid.write('tauvis='+str(self.w_tauvis.value)+'\n')
        fid.write('taunir='+str(self.w_taunir.value)+'\n')
        fid.write('taunir='+str(self.w_taunir.value)+'\n')
        fid.write('rsoilv='+str(self.w_rsoilv.value)+'\n')
        fid.write('rsoiln='+str(self.w_rsoiln.value)+'\n')

        fid.write('\n# Surface Properties\n')
        fid.write('Max_alpha_PT='+str(self.w_PT.value)+'\n')
        fid.write('x_LAD='+str(self.w_LAD.value)+'\n')
        fid.write('leaf_width='+str(self.w_leafwidth.value)+'\n')          
        fid.write('z0_soil='+str(self.w_zsoil.value)+'\n')
        fid.write('LANDCOVER='+str(self.w_lc.value)+'\n')

        fid.write('\n# Additional Options\n')
        fid.write('CalcG='+str(self.w_CalcG.value)+'\n')
        fid.write('G_constant='+str(self.w_Gconstant.value)+'\n')
        fid.write('G_ratio='+str(self.w_Gratio.value)+'\n')
        fid.write('GAmp='+str(self.w_GAmp.value)+'\n')
        fid.write('Gphase='+str(self.w_Gphase.value)+'\n')
        fid.write('Gshape='+str(self.w_Gshape.value)+'\n')
        fid.flush()
        fid.close()
        del fid
        print('Saved Configuration File')
                  
    def parseInputConfig(self,InputFile,isImage=False):
        ''' Parses the information contained in a configuration file into a dictionary''' 
        # look for the following variable names
        from re import match
        if isImage:
            input_vars=list(self.input_image_vars)
        else:
            input_vars=list(self.input_point_vars)
        # Add all the common variables
        for var in self.input_commom_vars:
            input_vars.append(var)
        # create the output configuration file:
        configdata=dict()
        for var in input_vars:
            configdata[var]=0
        try:
            fid=open(InputFile,'r')
        except IOError:
            print('Error reading ' + InputFile + ' file')
        for line in fid:
            
            if match('\s',line):# skip empty line
                continue
            elif line[0]=='#': #skip comment line
                continue
            elif '=' in line:
                string=line.split('#')[0].rstrip(' \r\n') # Remove comments in case they exist
                field,value=string.split('=')
                for var in input_vars:
                    if var==field:
                        configdata[field]=value
        del fid
        return configdata
           
    def SelectModel(self):
        ''' Widget to select the TSEB model'''
        import ipywidgets as widgets
        self.w_model=widgets.ToggleButtons(description='Select TSEB model to run:',
            options={'Priestley Taylor':'TSEB_PT', 'Dual-Time Difference':'DTD', 'Component Temperatures':'TSEB_2T'},
            value=self.model)
        
    def _on_model_change(self,name, value):
        '''Behaviour when TSEB model is changed'''
        if value=='DTD':
            self.w_Ta_0.visible=True
        else:
            self.w_Ta_0.visible=False
   

    def DefineSiteDescription(self):
        '''Widgets for site description parameters'''
        import ipywidgets as widgets

        self.w_lat=widgets.BoundedFloatText(value=self.lat,min=-90,max=90,description='Lat.',width=100)
        self.w_lon=widgets.BoundedFloatText(value=self.lon,min=-180,max=180,description='Lon.',width=100)
        self.w_alt= widgets.FloatText(value=self.alt,description='Alt.',width=100)
        
        self.w_stdlon= widgets.Dropdown(options=[str(-12+i) for i in range(25)],value='1',description='UTC:',width=100)
        self.w_zu= widgets.BoundedFloatText(value=self.zu,min=0.001,description='Wind meas. height',width=100)
        self.w_zt= widgets.BoundedFloatText(value=self.zt,min=0.001,description='T meas. height',width=100)
        self.sitePage=widgets.VBox([widgets.HBox([self.w_lat,self.w_lon,self.w_alt,self.w_stdlon]),
                    widgets.HBox([self.w_zu,self.w_zt])], background_color='#EEE')
        
    def SpectralProperties(self):
        '''Widgets for site spectral properties'''
        import ipywidgets as widgets
        self.w_rhovis=widgets.BoundedFloatText(value=self.rhovis,min=0,max=1,description='Leaf refl. PAR',width=80)
        self.w_tauvis=widgets.BoundedFloatText(value=self.tauvis,min=0,max=1,description='Leaf trans. PAR',width=80)
        self.w_rhonir=widgets.BoundedFloatText(value=self.rhonir,min=0,max=1,description='Leaf refl. NIR',width=80)
        self.w_taunir=widgets.BoundedFloatText(value=self.taunir,min=0,max=1,description='Leaf trans. NIR',width=80)
        
        self.w_rsoilv=widgets.BoundedFloatText(value=self.rsoilv,min=0,max=1,description='Soil refl. PAR',width=80)
        self.w_rsoiln=widgets.BoundedFloatText(value=self.rsoiln,min=0,max=1,description='Soil refl. NIR',width=80)
        self.w_emisVeg=widgets.BoundedFloatText(value=self.emisVeg,min=0,max=1,description='Leaf emissivity',width=80)
        self.w_emisSoil=widgets.BoundedFloatText(value=self.emisGrd,min=0,max=1,description='Soil emissivity',width=80)
        self.specPage=widgets.VBox([widgets.HBox([self.w_rhovis,self.w_tauvis,self.w_rhonir,self.w_taunir]),
                    widgets.HBox([self.w_rsoilv,self.w_rsoiln,self.w_emisVeg,self.w_emisSoil])], background_color='#EEE')
    
    def Meteorology(self):
        '''Widgets for meteorological forcing'''
        import ipywidgets as widgets
        self.w_DOY=widgets.BoundedFloatText(value=1,min=1,max=366,description='Day of Year',width=80)
        self.w_Time=widgets.BoundedFloatText(value=12,min=0,max=24,description='Dec. Time (h)',width=80)
        self.w_Ta_1=widgets.BoundedFloatText(value=0,min=-273.15,max=1e6,description='Tair (C)',width=80)
        self.w_Ta_0=widgets.BoundedFloatText(value=0,min=-273.15,max=1e6,description='Tair sunrise (C)',width=80)
        self.w_Ta_0.visible=False
        self.w_Sdn=widgets.BoundedFloatText(value=0,min=0,max=1e6,description='SW irradiance (W/m2)',width=80)
        self.w_u=widgets.BoundedFloatText(value=0.01,min=0.01,max=1e6,description='Windspeed (m/s)',width=80)
        self.w_ea=widgets.BoundedFloatText(value=0.0,min=0.0,max=1e6,description='Vapor Press. (mb)',width=80)
        metText=widgets.HTML('OPTIONAL: Leave empy to use estimated values for the following cells')
        self.w_Ldn=widgets.Text(value='',description='LW irradiance (W/m2)',width=80)
        self.w_p=widgets.Text(value='',description='Press. (mb)',width=80)
        self.metPage=widgets.VBox([widgets.HBox([self.w_DOY,self.w_Time,self.w_Ta_1,self.w_Ta_0]),
                                    widgets.HBox([self.w_Sdn,self.w_u,self.w_ea]),
                                    metText,
                                    widgets.HBox([self.w_Ldn,self.w_p])], background_color='#EEE')
         
    def _on_input_clicked(self,b):
        '''Behaviour when clicking the input file button'''
        self.w_inputtxt.value=str(self.GetInputFileName(title='Select Input Text File'))

    def _on_inputLST_clicked(self,b):
        '''Behaviour when clicking the LST input file button'''
        self.w_LSTtxt.value=str(self.GetInputFileName(title='Select Radiometric Temperature Image'))

    def _on_inputVZA_clicked(self,b):
        '''Behaviour when clicking the LST VZA file button'''
        self.w_VZAtxt.value=self.GetInputFileName(title='Select View Zenith Angle Image')

    def _on_inputLAI_clicked(self,b):
        '''Behaviour when clicking the LAI input file button'''
        self.w_LAItxt.value=self.GetInputFileName(title='Select Leaf Area Index Image')

    def _on_inputFc_clicked(self,b):
        '''Behaviour when clicking the Fc input file button'''
        self.w_Fctxt.value=self.GetInputFileName(title='Select Fractional Cover Image')

    def _on_inputFg_clicked(self,b):
        '''Behaviour when clicking the Fg input file button'''
        self.w_Fgtxt.value=self.GetInputFileName(title='Select Green Fraction Image')

    def _on_inputHc_clicked(self,b):
        '''Behaviour when clicking the Hc input file button'''
        self.w_Hctxt.value=self.GetInputFileName(title='Select Canopy Height Image')

    def _on_inputWc_clicked(self,b):
        '''Behaviour when clicking the Wc input file button'''
        self.w_Wctxt.value=self.GetInputFileName(title='Select Canopy Width/Height Ratio Image')

    def _on_inputmask_clicked(self,b):
        '''Behaviour when clicking the processing mask input file button'''
        self.w_masktxt.value=self.GetInputFileName(title='Select Image Mask')

    def GetInputFileName(self, title='Select Input File'):
        '''Creates a Tkinter input file dialog'''
        import sys
        # Import Tkinter GUI widgets
        if sys.version_info.major==2:
            from tkFileDialog import askopenfilename
            import Tkinter as tk
        else:
            from tkinter.filedialog import askopenfilename
            import tkinter as tk
        root=tk.Tk()
        root.withdraw()
        InputFile = askopenfilename(title=title) # show an "Open" dialog box and return the path to the selected file
        root.destroy() # Destroy the GUI
        if InputFile=='':return None
        return InputFile
    
    def _on_output_clicked(self,b):
        '''Behaviour when clicking the output file button'''
        self.w_outputtxt.value=self.GetOutputFileName()

    def GetOutputFileName(self, title='Select Output File'):
        '''Creates a Tkinter output file dialog'''
        import sys
        # Import Tkinter GUI widgets
        if sys.version_info.major==2:
            from tkFileDialog import asksaveasfilename
            import Tkinter as tk
        else:
            from tkinter.filedialog import asksaveasfilename
            import tkinter as tk
        root=tk.Tk()
        root.withdraw()
        OutputFile = asksaveasfilename(title=title) # show an "Open" dialog box and return the path to the selected file
        root.destroy()  # Destroy the GUI
        if OutputFile=='':return None
        return OutputFile
    
    def SurfaceProperties(self):
        '''Widgets for canopy properties'''
        import ipywidgets as widgets
        self.w_PT=widgets.BoundedFloatText(value=self.max_PT,min=0,description="Max. alphaPT",width=80)
        self.w_LAD=widgets.BoundedFloatText(value=self.x_LAD,min=0,description="LIDF param.",width=80)
        self.w_LAD.visible=False
        self.w_leafwidth=widgets.BoundedFloatText(value=self.leaf_width,min=0.001,description="Leaf width",width=80)
        self.w_zsoil=widgets.BoundedFloatText(value=self.z0soil,min=0,description="soil roughness",width=80)
        self.w_lc=widgets.Dropdown(options={'CROP':11,'GRASS':2,'SHRUB':5,'CONIFER':4,'BROADLEAVED':3},value=self.LANDCOVER,description="Land Cover Type",width=200)
        lcText=widgets.HTML(value='''Land cover information is used to estimate roughness. <BR>
                                    For shrubs, conifers and broadleaves we use the model of <BR>
                                    Schaudt & Dickinson (2000) Agricultural and Forest Meteorology. <BR>
                                    For crops and grasses we use a fixed ratio of canopy heigh''', width=100)

        self.vegPage=widgets.VBox([widgets.HBox([self.w_PT,self.w_LAD,self.w_leafwidth]),
                    widgets.HBox([self.w_zsoil,self.w_lc,lcText])], background_color='#EEE')
    
   
    def AdditionalOptionsPoint(self):
        '''Widgets for additional TSEB options'''
        import ipywidgets as widgets
        self.CalcGOptions()
        self.optPage=widgets.VBox([
            self.w_CalcG,
            self.w_Gratio,
            self.w_Gconstanttext,
            self.w_Gconstant,
            widgets.HBox([self.w_GAmp,self.w_Gphase,self.w_Gshape])], background_color='#EEE')

    def CalcGOptions(self):
        '''Widgets for method for computing soil heat flux'''
        import ipywidgets as widgets
        self.w_CalcG=widgets.ToggleButtons(description='Select method for soil heat flux',
            options={'Ratio of soil net radiation':1, 'Constant or measured value':0, 'Time dependent (Santanelo & Friedl)':2},value=self.CalcG, width=300)
        self.w_Gratio=widgets.BoundedFloatText(value=self.Gratio,min=0,max=1,description='G ratio (G/Rn)',width=80)
        self.w_Gconstant=widgets.FloatText(value=self.Gconstant,description='Value (W m-2)',width=80)
        self.w_Gconstant.visible=False
        self.w_Gconstanttext=widgets.HTML(value="Set G value (W m-2), ignored if G is present in the input file")
        self.w_Gconstanttext.visible=False
        self.w_Gconstant.visible=False
        self.w_GAmp=widgets.BoundedFloatText(value=self.GAmp,min=0,max=1,description='Amplitude (G/Rn)',width=80)
        self.w_GAmp.visible=False
        self.w_Gphase=widgets.BoundedFloatText(value=self.Gphase,min=-24,max=24,description='Time Phase (h)',width=80)
        self.w_Gphase.visible=False
        self.w_Gshape=widgets.BoundedFloatText(value=self.Gshape,min=0,max=24,description='Time shape (h)',width=80)
        self.w_Gshape.visible=False

    def _on_G_change(self,name, value):
        '''Behaviour when changing the soil heat flux model'''
        if value==0:
            self.w_Gratio.visible=False
            self.w_Gconstant.visible=True
            self.w_Gconstanttext.visible=True
            self.w_GAmp.visible=False
            self.w_Gphase.visible=False
            self.w_Gshape.visible=False
        elif value==1:
            self.w_Gratio.visible=True
            self.w_Gconstant.visible=False
            self.w_Gconstanttext.visible=False
            self.w_GAmp.visible=False
            self.w_Gphase.visible=False
            self.w_Gshape.visible=False
        elif value==2:
            self.w_Gratio.visible=False
            self.w_Gconstant.visible=False
            self.w_Gconstanttext.visible=False
            self.w_GAmp.visible=True
            self.w_Gphase.visible=True
            self.w_Gshape.visible=True

    def GetDataTSEBWidgets(self,isImage):
        '''Parses the parameters in the GUI to TSEB variables for running TSEB'''
        self.TSEB_MODEL=self.w_model.value
        self.lat,self.lon,self.alt,self.stdlon,self.z_u,self.z_t=(self.w_lat.value,
                self.w_lon.value,self.w_alt.value,float(self.w_stdlon.value)*15,
                self.w_zu.value,self.w_zt.value)
        
        self.emisVeg,self.emisGrd=self.w_emisVeg.value,self.w_emisSoil.value
        self.spectraVeg={'rho_leaf_vis':self.w_rhovis.value, 'tau_leaf_vis':self.w_tauvis.value,
                    'rho_leaf_nir':self.w_rhonir.value, 'tau_leaf_nir':self.w_taunir.value }
        self.spectraGrd={'rsoilv':self.w_rsoilv.value, 'rsoiln':self.w_rsoiln.value}
        self.Max_alpha_PT, self.x_LAD, self.leaf_width,self.z0_soil,self.LANDCOVER=(self.w_PT.value,self.w_LAD.value,
               self.w_leafwidth.value,self.w_zsoil.value,self.w_lc.value)
       
        if self.w_CalcG.value==0:
            self.CalcG=[0,self.w_Gconstant.value]
        elif self.w_CalcG.value==1:
            self.CalcG=[1,self.w_Gratio.value]
        elif self.w_CalcG.value==2:
            self.CalcG=[2,[12.0,self.w_GAmp.value,self.w_Gphase.value,self.w_Gshape.value]]
        self.OutputFile=self.w_outputtxt.value
        
        if isImage:
            # Get all the input parameters
            self.input_LST=self.w_LSTtxt.value
            self.input_VZA=self.w_VZAtxt.value
            self.input_LAI=self.w_LAItxt.value
            self.input_Hc=self.w_Hctxt.value
            self.input_Fc=self.w_Fctxt.value
            self.input_Fg=self.w_Fgtxt.value
            self.input_Wc=self.w_Wctxt.value
            self.input_mask=self.w_masktxt.value
            
            self.DOY,self.Time,self.Ta_1,self.Sdn,self.u,self.ea,self.Ldn,self.p=(self.w_DOY.value,self.w_Time.value,
                    self.w_Ta_1.value,self.w_Sdn.value,self.w_u.value,self.w_ea.value,self.w_Ldn.value,self.w_p.value)
            if self.TSEB_MODEL=='DTD':
                self.Ta_0=self.w_Ta_0.value
        else:
            self.InputFile=self.w_inputtxt.value

    def GetDataTSEB(self,configdata,isImage):
        '''Parses the parameters in a configuration file directly to TSEB variables for running TSEB'''
        self.TSEB_MODEL=configdata['TSEB_MODEL']
        self.lat,self.lon,self.alt,self.stdlon,self.z_u,self.z_t=(float(configdata['lat']),
                float(configdata['lon']),float(configdata['altitude']),float(configdata['stdlon']),
                float(configdata['z_u']),float(configdata['z_t']))
      
        self.emisVeg,self.emisGrd=float(configdata['emisVeg']),float(configdata['emisGrd'])
        self.spectraVeg={'rho_leaf_vis':float(configdata['rhovis']), 'tau_leaf_vis':float(configdata['tauvis']),
                    'rho_leaf_nir':float(configdata['rhonir']), 'tau_leaf_nir':float(configdata['taunir'])}
        self.spectraGrd={'rsoilv':float(configdata['rsoilv']), 'rsoiln':float(configdata['rsoiln'])}
        self.Max_alpha_PT, self.x_LAD, self.leaf_width,self.z0_soil,self.LANDCOVER=(float(configdata['Max_alpha_PT']),
                        float(configdata['x_LAD']),float(configdata['leaf_width']),float(configdata['z0_soil']),int(configdata['LANDCOVER']))
        
        if int(configdata['CalcG'])==0:
            self.CalcG=[0,float(configdata['G_constant'])]
        elif int(configdata['CalcG'])==1:
            self.CalcG=[1,float(configdata['G_ratio'])]
        elif int(configdata['CalcG'])==2:
            self.CalcG=[2,[12.0,float(configdata['GAmp']),float(configdata['Gphase']),float(configdata['Gshape'])]]

        self.OutputFile=configdata['OutputFile']
        
        if isImage:
            # Get all the input parameters
            self.input_LST=str(configdata['Input_LST']).strip('"')
            self.input_VZA=str(configdata['Input_VZA']).strip('"')
            self.input_LAI=str(configdata['Input_LAI']).strip('"')
            self.input_Hc=str(configdata['Input_Hc']).strip('"')
            self.input_Fc=str(configdata['Input_Fc']).strip('"')
            self.input_Fg=str(configdata['Input_Fg']).strip('"')
            self.input_Wc=str(configdata['Input_Wc']).strip('"')
            self.input_mask=str(configdata['USE_MASK']).strip('"')
            self.DOY,self.Time,self.Ta_1,self.Sdn,self.u,self.ea,self.Ldn,self.p=(float(configdata['DOY']),float(configdata['Time']),
                float(configdata['Ta_1']),float(configdata['Sdn']),float(configdata['u']),float(configdata['ea']),
                str(configdata['Ldn']).strip('"'),str(configdata['p']).strip('"'))
            if self.TSEB_MODEL=='DTD':
                self.Ta_0=float(configdata['Ta_0'])
        else:
            self.InputFile=str(configdata['InputFile']).strip('"')
    
    def RunTSEBLocalImage(self):
        ''' Runs TSEB for all the pixel in an image'''
        import TSEB
        import numpy as np
        import gdal
        import ipywidgets as widgets
        from IPython.display import display
        from os.path import splitext, dirname, exists
        from os import mkdir
        # Create an input dictionary
        inDataArray=dict()
        # Open the LST data according to the model
        if self.TSEB_MODEL=='TSEB_PT' or self.TSEB_MODEL=='DTD':
            try:
                # Read the image mosaic and get the LST at noon time
                fid=gdal.Open(self.input_LST,gdal.GA_ReadOnly)
                lst=fid.GetRasterBand(1).ReadAsArray()
                inDataArray['T_R1']=lst
                dims=np.shape(lst)
                # Get the spatial reference information
                prj=fid.GetProjection()
                geo=fid.GetGeoTransform()
                del lst
            except:
                print('Error reading LST file '+self.input_LST)
                return
            if self.TSEB_MODEL=='DTD':# Also read the sunrise LST
                # Read the image mosaic and get the LST at sunrise time
                try:
                    inDataArray['T_R0']=fid.GetRasterBand(2).ReadAsArray()
                except:
                    print('Error reading sunrise LST file '+str(self.input_LST))
                    return
            del fid
        elif self.TSEB_MODEL=='TSEB_2T':
            try:
                # Read the Noon Component Temperatures
                fid=gdal.Open(self.input_LST,gdal.GA_ReadOnly)
                inDataArray['T_C']=fid.GetRasterBand(1).ReadAsArray()
                inDataArray['T_S']=fid.GetRasterBand(2).ReadAsArray()
                dims=np.shape(inDataArray['T_C'])
                # Get the spatial reference information
                prj=fid.GetProjection()
                geo=fid.GetGeoTransform()
                del fid
            except:
                print('Error reading Component Temperatures file '+str(self.input_LST))
                return
        # Read the image mosaic and get the LAI
        success,inDataArray['LAI']=self.OpenGDALImage(self.input_LAI,dims,'Leaf Area Index')
        if not success: return
        # Read the image View Zenith Angle
        success,inDataArray['VZA']=self.OpenGDALImage(self.input_VZA,dims,'View Zenith Angle')
        if not success: return
        # Read the fractional cover data
        success,inDataArray['f_C']=self.OpenGDALImage(self.input_Fc,dims,'Fractional Cover')
        if not success: return
        # Read the Canopy Height data
        success,inDataArray['h_C']=self.OpenGDALImage(self.input_Hc,dims,'Canopy Height')
        if not success:return
        # Read the canopy witdth ratio
        success,inDataArray['w_C']=self.OpenGDALImage(self.input_Wc,dims,'Canopy Width Ratio')
        if not success: return
        # Read the Green fraction
        success,inDataArray['f_g']=self.OpenGDALImage(self.input_Fg,dims,'Green Fraction')
        if not success: return
        #Calculate illumination conditions
        sza,saa=TSEB.met.Get_SunAngles(self.lat,self.lon,self.stdlon,self.DOY,self.Time)
        # Estimation of diffuse radiation
        try:
            p=float(self.p)
        except:
            p=TSEB.met.CalcPressure(self.alt)
        difvis,difnir, fvis,fnir=TSEB.rad.CalcDifuseRatio(self.Sdn,sza,press=p)
        Skyl=difvis*fvis+difnir*fnir
        Sdn_dir=self.Sdn*(1.0-Skyl)
        Sdn_dif=self.Sdn*Skyl
        # incoming long wave radiation
        Ta_K_1=self.Ta_1+273.15
        if self.TSEB_MODEL=='DTD':
            Ta_K_0=self.Ta_0+273.15
        try:
            Lsky=float(self.Ldn)
        except:
            emisAtm = TSEB.rad.CalcEmiss_atm(self.ea,Ta_K_1)
            Lsky = emisAtm * TSEB.met.CalcStephanBoltzmann(Ta_K_1)
        # Create the output dictionary
        outputStructure=self.getOutputStructure()
        tsebout=dict()
        for field in outputStructure:tsebout[field]=np.zeros(dims)
        # Open the processing maks and get the id for the cells to process
        if self.input_mask!='0':
            try:
                fid=gdal.Open(self.input_mask,gdal.GA_ReadOnly)
                mask=fid.GetRasterBand(1).ReadAsArray()
                del fid
            except:
                print('ERROR: file read '+ str(self.input_mask) + '\n Please type a valid mask file name or USE_MASK=0 for processing the whole image')
        else:
            mask=np.ones(dims)
        ids=np.where(mask>0)
        # Create a progress bar widget
        try:
            f=widgets.FloatProgress(min=0,max=len(ids[0]),step=1,description='Progress:')
            display(f)
            isWidget=True
        except:
            progress_bar=[(i+1)*5 for i in range(20)]
            progress_id=0
            print('0% processed')
            isWidget=False
        # Loop and run TSEB for all the valid cells
        for i,row in enumerate(ids[0]):
            col=ids[1][i]
            if isWidget:
                f.value=i
            else:
                progress=100*float(i)/len(ids[0])
                if progress>=progress_bar[progress_id]:
                    print(str(progress_bar[progress_id])+ '% ')
                    progress_id=progress_id+1
            # Get the LAI
            lai=inDataArray['LAI'][row,col]
            fc=inDataArray['f_C'][row,col]
            f_g=inDataArray['f_g'][row,col]
            if fc<=0.01 or lai<=0 or np.isnan(lai):
                lai=0
                fc=0
            wc=inDataArray['w_C'][row,col]
            hc=inDataArray['h_C'][row,col]
            # Calculate Roughness
            z_0M, d_0=TSEB.res.CalcRoughness (lai, hc,wc,self.LANDCOVER)
            vza=inDataArray['VZA'][row,col]
            if self.TSEB_MODEL=='DTD':
                #Run DTD
                Tr_K_0=inDataArray['T_R0'][row,col]+273.15
                Tr_K_1=inDataArray['T_R1'][row,col]+273.15
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,Ri,n_iterations]=[0]*20
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,Ri,
                     n_iterations]=TSEB.DTD(Tr_K_0,Tr_K_1,vza,Ta_K_0,Ta_K_1,self.u,self.ea,p,Sdn_dir,Sdn_dif,fvis,fnir,
                        sza,Lsky,lai,hc,self.emisVeg,self.emisGrd,self.spectraVeg,self.spectraGrd,z_0M,d_0,self.zu,self.zt,
                        f_c=fc,wc=wc,f_g=f_g,leaf_width=self.leaf_width,z0_soil=self.z0_soil,alpha_PT=self.Max_alpha_PT,
                        CalcG=self.CalcG)
            elif self.TSEB_MODEL=='TSEB_PT':        
                #Run TSEB
                Tr_K_1=inDataArray['T_R1'][row,col]+273.15
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,n_iterations]=[0]*19
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,
                     n_iterations]=TSEB.TSEB_PT(Tr_K_1,vza,Ta_K_1,self.u,self.ea,p,Sdn_dir,Sdn_dif,fvis,fnir,sza,Lsky,lai,
                        hc,self.emisVeg,self.emisGrd,self.spectraVeg,self.spectraGrd,z_0M,d_0,self.zu,self.zt,
                        f_c=fc,f_g=f_g,wc=wc,leaf_width=self.leaf_width,z0_soil=self.z0_soil,alpha_PT=self.Max_alpha_PT,
                        CalcG=self.CalcG)
            elif self.TSEB_MODEL=='TSEB_2T':
                Tc=inDataArray['T_C'][row,col]+273.15
                Ts=inDataArray['T_S'][row,col]+273.15
                #Run TSEB with Component Temperature
                [flag, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,counter]=[0]*17
                if lai==0:# Bare Soil, One Source Energy Balance Model
                    z_0M=self.z0_soil
                    d_0=5.*z_0M
                    albedoGrd=fvis*self.spectraGrd['rsoilv']+fnir* self.spectraGrd['rsoiln']
                    [flag,S_nS, L_nS, LE_S,H_S,G,R_a,u_friction, L,counter]=TSEB.OSEB(Ts,Ta_K_1,self.u,self.ea,p,self.Sdn,
                                Lsky,self.emisGrd,albedoGrd,z_0M,d_0,self.zu,self.zt, CalcG=self.CalcG)
                else:
                    F=lai/fc# Get the local LAI and compute the clumping index
                    omega0=TSEB.CI.CalcOmega0_Kustas(lai, fc, isLAIeff=True)
                    Omega=TSEB.CI.CalcOmega_Kustas(omega0,sza,wc=self.wc)
                    LAI_eff=F*Omega
                    # Estimate the net shorwave radiation 
                    S_nS, S_nC = TSEB.rad.CalcSnCampbell (LAI_eff, sza, Sdn_dir, Sdn_dif, fvis,fnir, 
                        self.spectraVeg['rho_leaf_vis'], self.spectraVeg['tau_leaf_vis'],
                        self.spectraVeg['rho_leaf_nir'], self.spectraVeg['tau_leaf_nir'], 
                        self.spectraGrd['rsoilv'], self.spectraGrd['rsoiln'])
                    # And the net longwave radiation
                    L_nS,L_nC=TSEB.rad.CalcLnKustas (Tc, Ts,Lsky, lai,self.emisVeg, self.emisGrd)
                    # Run TSEB with the component temperatures Ts and Tc    
                    [flag,T_AC,LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L, n_iterations]=TSEB.TSEB_2T(Tc, Ts,Ta_K_1,
                        self.u,self.ea,p,S_nS, S_nC, L_nS,L_nC,lai,hc,z_0M, d_0, self.zu,self.zt,leaf_width=self.leaf_width,f_c=fc,
                         z0_soil=self.z0_soil,alpha_PT=self.Max_alpha_PT,
                        CalcG=self.CalcG)
            
            # Calculate the bulk fluxes
            LE=LE_C+LE_S
            H=H_C+H_S
            Rn=S_nC+S_nS+L_nC+L_nS
            # Write the data in the output dictionary
            tsebout['R_A1'][row,col]=R_a
            tsebout['R_X1'][row,col]=R_x
            tsebout['R_S1'][row,col]=R_s
            tsebout['R_n1'][row,col]=S_nS+S_nC+L_nS+L_nC
            tsebout['R_ns1'][row,col]=S_nS+S_nC
            tsebout['R_nl1'][row,col]=L_nS+L_nC
            tsebout['delta_R_n1'][row,col]=S_nC+L_nC
            tsebout['H_C1'][row,col]=H_C
            tsebout['H_S1'][row,col]=H_S
            tsebout['H1'][row,col]=H
            tsebout['G1'][row,col]=G
            tsebout['LE_C1'][row,col]=LE_C
            tsebout['LE_S1'][row,col]=LE_S
            tsebout['LE1'][row,col]=LE
            if tsebout['LE1'][row,col]>0:
                tsebout['LE_partition'][row,col]=LE_C/LE
            tsebout['T_C1'][row,col]=Tc
            tsebout['T_S1'][row,col]=Ts
            tsebout['T_AC1'][row,col]=T_AC
            tsebout['L'][row,col]=L
            tsebout['u_friction'][row,col]=u_friction
            tsebout['theta_s1'][row,col]=sza
            tsebout['albedo1'][row,col]=1.0-Rn/self.Sdn
            tsebout['F'][row,col]=lai
        # Write the TIFF output
        outdir=dirname(self.OutputFile)
        if not exists(outdir):
            mkdir(outdir)
        self.WriteTifOutput(self.OutputFile,tsebout, geo, prj,self.fields)
        outputfile=splitext(self.OutputFile)[0]+'_ancillary.tif'
        self.WriteTifOutput(outputfile,tsebout, geo, prj,self.anc_fields)
        print('Saved Files')

    def OpenGDALImage(self,inputString,dims,variable):
        '''Open a GDAL image and returns and array with its first band'''
        import gdal
        import numpy as np
        success=True
        try:
            array=np.zeros(dims)+float(inputString)
        except:
            try:
                fid=gdal.Open(inputString,gdal.GA_ReadOnly)
                array=fid.GetRasterBand(1).ReadAsArray()
                del fid
            except:
                print('ERROR: file read '+ str(inputString) + '\n Please type a valid file name or a numeric value for ' +variable)
                return False,False
        return success,array

    def WriteTifOutput(self,outfile,output,geo, prj, fields):
        '''Writes the arrays of an output dictionary which keys match the list in fields to a GeoTIFF '''
        import gdal
        import numpy as np
        rows,cols=np.shape(output['H1'])
        driver = gdal.GetDriverByName('GTiff')
        nbands=len(fields)
        ds = driver.Create(outfile, cols, rows, nbands, gdal.GDT_Float32)
        ds.SetGeoTransform(geo)
        ds.SetProjection(prj)
        for i,field in enumerate(fields):
            band=ds.GetRasterBand(i+1)
            band.SetNoDataValue(0)
            band.WriteArray(output[field])
            band.FlushCache()
        ds.FlushCache()
        del ds

    def getOutputStructure(self):
        ''' Output fields in TSEB'''        
        outputStructure = (
        # resistances
        'R_A1',   #resistance to heat transport in the surface layer (s/m) at time t1
        'R_X1',  #resistance to heat transport in the canopy surface layer (s/m) at time t1
        'R_S1',   #resistance to heat transport from the soil surface (s/m) at time t1
        # fluxes
        'R_n1',   #net radiation reaching the surface at time t1
        'R_ns1',  #net shortwave radiation reaching the surface at time t1
        'R_nl1',  #net longwave radiation reaching the surface at time t1 
        'delta_R_n1',#net radiation divergence in the canopy at time t1
        'H_C1',   #canopy sensible heat flux (W/m^2) at time t1
        'H_S1',   #soil sensible heat flux (W/m^2) at time t1
        'H1',     #total sensible heat flux (W/m^2) at time t1
        'G1',     #ground heat flux (W/m^2) at time t1
        'LE_C1',    #canopy latent heat flux (W/m^2) at time t1
        'LE_S1',    #soil latent heat flux (W/m^2) at time t1
        'LE1',    #total latent heat flux (W/m^2) at time t1
        'LE_partition',   #Latent Heat Flux Partition (LEc/LE) at time t1
        # temperatures (might not be accurate)
        'T_C1',   #canopy temperature at time t1 (deg C)	
        'T_S1',   #soil temperature at time t1 (deg C)
        'T_AC1',   #air temperature at the canopy interface at time t1 (deg C)
        # miscaleneous
        'albedo1',    # surface albedo (Rs_out/Rs_in)
        'omega0', #nadir view vegetation clumping factor
        'alpha',  #the priestly Taylor factor
        'Ri',     #Richardson number at time t1
        'L',      #Monin Obukhov Length at time t1
        'u_friction',# Friction velocity
        'theta_s1',#Sun zenith angle at time t1
        'F')      # Leaf Area Index
        
        return outputStructure
            
    def CheckDataPointSeriers(self):
        '''Checks that all the data required for TSEB is contained in an input ascci table'''
        success=False
        # Mandatory Input Fields
        MandatoryFields_TSEB_PT=('Year','DOY','Time','Trad','VZA','Ta','u','ea','Sdn','LAI','hc')
        MandatoryFields_DTD=('Year','DOY','Time','Trad_0','Trad','VZA','Ta_0','Ta','u','ea','Sdn','LAI','hc')                        
        MandatoryFields_TSEB_2T=('Year','DOY','Time','Tc','Ts','Ta','u','ea','Sdn','LAI','hc')  
        # Check that all mandatory input variables exist
        if self.TSEB_MODEL=='TSEB_PT':
            for field in MandatoryFields_TSEB_PT:
                if field not in self.inputNames:
                    print('ERROR: ' +field +' not found in file '+ self.InputFile)
                    return success
        elif self.TSEB_MODEL=='DTD':
            for field in MandatoryFields_DTD:
                if field not in self.inputNames:
                    print('ERROR: ' +field +' not found in file '+ self.InputFile)
                    return success
        elif self.TSEB_MODEL=='TSEB_2T':
            for field in MandatoryFields_TSEB_2T:
                if field not in self.inputNames:
                    print('ERROR: ' +field +' not found in file '+ self.InputFile)
                    return success
        else:
            print('Not valid TSEB model, check your configuration file')
            return success
        return True
    
    def RunTSEBPointSeries(self):
        ''' Runs TSEB for all the pixel in an image'''
        import TSEB
        import ipywidgets as widgets
        from IPython.display import display
        from  os. path import dirname, exists
        from os import mkdir
        # Output Headers
        outputTxtFieldNames = ['Year', 'DOY', 'Time','LAI','f_g', 'skyl', 'VZA', 
                               'SZA', 'SAA','L_sky','Rn_model','Rn_sw_veg', 'Rn_sw_soil', 
                               'Rn_lw_veg', 'Rn_lw_soil', 'Tc', 'Ts', 'Tac', 
                               'LE_model', 'H_model', 'LE_c', 'H_c', 'LE_s', 'H_s', 
                               'flag', 'zo', 'd', 'G_model', 'R_s', 'R_x', 'R_a', 
                               'u_friction', 'L',  'n_iterations']
        # Create and open the ouput file
        outdir=dirname(self.OutputFile)
        if not exists(outdir):
            mkdir(outdir)
        fid_out=open(self.OutputFile,'w')
        header=outputTxtFieldNames[0]
        # Write the ouput headers
        n_headers=len(outputTxtFieldNames)
        for sds in range(1,n_headers):
            header=header+'\t'+outputTxtFieldNames[sds]
        fid_out.write(header+'\n')
        # Open the input file
        try:
            num_lines = sum(1 for line in open(self.InputFile,'r'))
            infid=open(self.InputFile,'r')
            # Read the first line of the file headers
            self.inputNames=infid.readline().rstrip(' \n\r')
            self.inputNames=self.inputNames.split('\t')
        except IOError:
            print('Error reading input file : '+self.InputFile)
            return
        # Check that the input file contains all the needed variables
        success=self.CheckDataPointSeriers()
        if not success:return
        # Create a progress bar widget
        try:
            f=widgets.FloatProgress(min=0,max=num_lines-1,step=1,description='Progress:')
            display(f)
            isWidget=True
        except:
            progress_bar=[(i+1)*5 for i in range(20)]
            progress_id=0
            print('0% processed')
            isWidget=False
        # Loop all the lines in the tables
        for i,indata in enumerate(infid): # Read on line at a time
            if isWidget:
                f.value=i # Update the progres bar until it reaches the last line =100%
            else:
                progress=100*float(i)/(num_lines-1)
                if progress>=progress_bar[progress_id]:
                    print(str(progress_bar[progress_id])+ '% ')
                    progress_id=progress_id+1
            indata=indata.strip(' \n\r') # remove the carriage returns at the end of the line
            indata=indata.split('\t') # Split the string by tabs to have a list of values
            # Get the Year, Doy of the Year and Time: i.e.: finds the column where 'Year' is placed and get the value from indata
            Year,DOY,Time=(float(indata[self.inputNames.index('Year')]),float(indata[self.inputNames.index('DOY')]),float(indata[self.inputNames.index('Time')]))
            #Calculate illumination conditions if not included in the input file
            if 'SZA' not in self.inputNames:
                sza,dummy=TSEB.met.Get_SunAngles(self.lat,self.lon,self.stdlon,DOY,Time)
            else:
                sza=float(indata[self.inputNames.index('SZA')])
            if 'SAA' not in self.inputNames:
                dummy,saa=TSEB.met.Get_SunAngles(self.lat,self.lon,self.stdlon,DOY,Time)
            else:
                saa=float(indata[self.inputNames.index('SAA')])
            # Estimation of diffuse radiation
            if 'p' not in self.inputNames: # Estimate barometric pressure from the altitude if not included in the table
                p=TSEB.met.CalcPressure(self.alt)
            else:
                p=float(indata[self.inputNames.index('p')])
            # Esimate diffuse and direct irradiance
            Sdn=float(indata[self.inputNames.index('Sdn')])# Get the solar irradiance value
            difvis,difnir, fvis,fnir=TSEB.rad.CalcDifuseRatio(Sdn,sza,press=p)
            Skyl=difvis*fvis+difnir*fnir
            Sdn_dir=Sdn*(1.0-Skyl) # Direct irradiance
            Sdn_dif=Sdn*Skyl # Diffuse irradiance
            # Get the temperatures
            if self.TSEB_MODEL=='TSEB_PT' or self.TSEB_MODEL=='DTD':# Radiometric composite temperature
                Ta_K_1=float(indata[self.inputNames.index('Ta')])+273.15
                Tr_K_1=float(indata[self.inputNames.index('Trad')])+273.15
                if self.TSEB_MODEL=='DTD': # Get the near sunrise temperatures
                    Ta_K_0=float(indata[self.inputNames.index('Ta_0')])+273.15
                    Tr_K_0=float(indata[self.inputNames.index('Trad_0')])+273.15
            elif self.TSEB_MODEL=='TSEB_2T': # one air temperature observation amd two component temperatures
                Ta_K_1=float(indata[self.inputNames.index('Ta')])+273.15
                Tc=float(indata[self.inputNames.index('Tc')])+273.15
                Ts=float(indata[self.inputNames.index('Ts')])+273.15

            # incoming long wave radiation
            ea=float(indata[self.inputNames.index('ea')])
            u=float(indata[self.inputNames.index('u')])
            if 'Ldn' not in self.inputNames: # Check if downwelling LW radiance is in the table
                # Calculate downwelling LW radiance otherwise
                emisAtm = TSEB.rad.CalcEmiss_atm(ea,Ta_K_1)
                Lsky = emisAtm * TSEB.met.CalcStephanBoltzmann(Ta_K_1)
            else:
                Lsky=float(indata[self.inputNames.index('Ldn')])
            # Get the LAI and canopy height
            lai=float(indata[self.inputNames.index('LAI')])
            hc=float(indata[self.inputNames.index('hc')])
            # Get the other canopy parameters if present in the input table otherwise use defaults
            if 'fc' not in self.inputNames: # Fractional cover
                fc=self.f_c
            else:
                fc=float(indata[self.inputNames.index('fc')])
            if 'wc' not in self.inputNames: # Canopy width to height ratio
                wc=self.wc
            else:
                wc=float(indata[self.inputNames.index('wc')])
            if 'fg' not in self.inputNames: # Green fraction
                f_g=self.f_g
            else:
                f_g=float(indata[self.inputNames.index('fg')])
            # Calculate Roughness
            z_0M, d_0=TSEB.res.CalcRoughness (lai, hc,wc,self.LANDCOVER)
            vza=float(indata[self.inputNames.index('VZA')])
            # get the Soil Heat flux if CalcG includes the option of measured G
            if self.CalcG[0]==0: # Constant G
                if 'G' in self.inputNames:
                    self.CalcG[1]=float(indata[self.inputNames.index('G')])
            elif self.CalcG[0]==2: # Santanello and Friedls G
                self.CalcG[1][0]=Time # Set the time in the CalcG flag to compute the Santanello and Friedl G
            if self.TSEB_MODEL=='DTD':
                #Run DTD
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,Ri,n_iterations]=[0]*20
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,Ri,
                     n_iterations]=TSEB.DTD(Tr_K_0,Tr_K_1,vza,Ta_K_0,Ta_K_1,u,ea,p,Sdn_dir,Sdn_dif,fvis,fnir,
                        sza,Lsky,lai,hc,self.emisVeg,self.emisGrd,self.spectraVeg,self.spectraGrd,z_0M,d_0,self.zu,self.zt,
                        f_c=fc,wc=wc,f_g=f_g,leaf_width=self.leaf_width,z0_soil=self.z0_soil,alpha_PT=self.Max_alpha_PT,
                        CalcG=self.CalcG)
            elif self.TSEB_MODEL=='TSEB_PT':        
                #Run TSEB
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,n_iterations]=[0]*19
                [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,
                     n_iterations]=TSEB.TSEB_PT(Tr_K_1,vza,Ta_K_1,u,ea,p,Sdn_dir,Sdn_dif,fvis,fnir,sza,Lsky,lai,
                        hc,self.emisVeg,self.emisGrd,self.spectraVeg,self.spectraGrd,z_0M,d_0,self.zu,self.zt,
                        f_c=fc,f_g=f_g,wc=wc,leaf_width=self.leaf_width,z0_soil=self.z0_soil,alpha_PT=self.Max_alpha_PT,
                        CalcG=self.CalcG)
            elif self.TSEB_MODEL=='TSEB_2T':
                #Run TSEB with Component Temperature
                [flag, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,counter]=[0]*17
                if lai==0:# Bare Soil, One Source Energy Balance Model
                    z_0M=self.z0_soil
                    d_0=5.*z_0M
                    albedoGrd=fvis*self.spectraGrd['rsoilv']+fnir* self.spectraGrd['rsoiln']
                    [flag,S_nS, L_nS, LE_S,H_S,G,R_a,u_friction, L,counter]=TSEB.OSEB(Ts,Ta_K_1,u,ea,p,Sdn,
                                Lsky,self.emisGrd,albedoGrd,z_0M,d_0,self.zu,self.zt, CalcG=self.CalcG)
                else:
                    F=lai/fc# Get the local LAI and compute the clumping index
                    omega0=TSEB.CI.CalcOmega0_Kustas(lai, fc,isLAIeff=True)
                    Omega=TSEB.CI.CalcOmega_Kustas(omega0,sza,wc=wc)
                    LAI_eff=F*Omega
                    # Estimate the net shorwave radiation 
                    S_nS, S_nC = TSEB.rad.CalcSnCampbell (LAI_eff, sza, Sdn_dir, Sdn_dif, fvis,fnir, 
                        self.spectraVeg['rho_leaf_vis'], self.spectraVeg['tau_leaf_vis'],
                        self.spectraVeg['rho_leaf_nir'], self.spectraVeg['tau_leaf_nir'], 
                        self.spectraGrd['rsoilv'], self.spectraGrd['rsoiln'])
                    # And the net longwave radiation
                    L_nS,L_nC=TSEB.rad.CalcLnKustas (Tc, Ts,Lsky, lai,self.emisVeg, self.emisGrd)
                    # Run TSEB with the component temperatures Ts and Tc    
                    [flag,T_AC,LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L, n_iterations]=TSEB.TSEB_2T(Tc, Ts,Ta_K_1,
                        u,ea,p,S_nS, S_nC, L_nS,L_nC,lai,hc,z_0M, d_0, self.zu,self.zt,leaf_width=self.leaf_width,f_c=fc,
                         z0_soil=self.z0_soil,alpha_PT=self.Max_alpha_PT,
                         CalcG=self.CalcG)
            # Calculate the bulk fluxes
            LE=LE_C+LE_S
            H=H_C+H_S
            Rn=S_nC+S_nS+L_nC+L_nS
                # Write the results in the output file
            #    'Year', 'DOY', 'Time','LAI','f_g', 'skyl', 'VZA', 
            #    'SZA', 'SAA','L_sky','Rn_model','Rn_sw_veg', 'Rn_sw_soil', 
            #    'Rn_lw_veg', 'Rn_lw_soil', 'Tc', 'Ts', 'Tac', 
            #    'LE_model', 'H_model', 'LE_c', 'H_c', 'LE_s', 'H_s', 
            #    'flag', 'zo', 'd', 'G_model', 'R_s', 'R_x', 'R_a', 
            #    'u_friction', 'L',  'n_iterations'
            fid_out.write(str(Year)+'\t'+str(DOY)+'\t'+str(Time)+'\t'+str(lai)+'\t'+str(f_g)+'\t'+str(Skyl)+
                          '\t'+str(vza)+'\t'+str(sza)+'\t'+str(saa)+'\t'+str(Lsky)+'\t'+str(Rn)+'\t'+str(S_nC)+
                          '\t'+str(S_nS)+'\t'+str(L_nC)+'\t'+str(L_nS)+'\t'+str(Tc)+'\t'+str(Ts)+'\t'+str(T_AC)+
                          '\t'+str(LE)+'\t'+str(H)+'\t'+str(LE_C)+'\t'+str(H_C)+'\t'+str(LE_S)+'\t'+str(H_S)+
                          '\t'+str(flag)+'\t'+str(z_0M)+'\t'+str(d_0)+'\t'+str(G)+'\t'+str(R_s)+'\t'+str(R_x)+
                          '\t'+str(R_a)+'\t'+str(u_friction)+'\t'+str(L)+'\t'+str(n_iterations)+'\n')
            fid_out.flush()
        # Close all the files
        fid_out.close()
        infid.close()
        print('Done')
