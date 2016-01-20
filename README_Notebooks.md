# Use of the Notebooks' GUI for running pyTSEB

## Prepare the data

### ProcessLocalImage.ipynb
#### Main input image
The main input data in this case is an image containing the radiometric temperature(s). Any [GDAL compatible raster](http://www.gdal.org/formats_list.html) should work except those images based on datasets (i.e. HDF, netCDF, etc.). The input temperature data should be in Celsius. Depending on the TSEB model to use you have to be sure to include in the file the following channels

* Priestley-Taylor TSEB (**TSEB-PT**)
	1. Radiometric surface temperature

* Dual-Time Difference TSEB (**DTD**)
	1. Radiometric surface temperature around noon
	2. Radiometric surface temperature around sunrise

* Composite temperatures TSEB( **TSEB-2T**)
	1. Canopy temperature
	2. Soil temperature

#### Ancillary input
The following variables can be provided either in an image format, in which their size should exactly match the size of the temperature images, or a sigle constant value to use in the whole scene

* Effective Leaf Area Index
* View Zenith Angle (degrees)
* Fractional cover, fc (0-1)
* Canopy height (m)
* Canopy Width to Height ratio (w/hc)
* Green fraction, fg (0-1)

The Processing mask is also optional, if provided only those (cloud-free?) pixels with a mask value >0 will be processed

### ProcessPointTimeSeries.ipynb
All the input information in the point time series versions must be provided through an ASCCI (text) table. The table should contain in the first row the headers. Use tabs ("\t") to separate the columns. The table should contain at least the following field names (case sensitive):

>* **TSEB-PT** : `Year, DOY, Time, Trad, VZA, Ta, u, ea, Sdn, LAI & hc`
>* **DTD** : `Year, DOY, Time, Trad_0, Trad, VZA, Ta_0, Ta, u, ea, Sdn, LAI & hc`
>* **TSEB-2T** : `Year, DOY, Time, Tc, Ts, Ta, u, ea, Sdn, LAI & hc`

The order of the columns is not relevant, and neither whethere there are additional columns in the table (they will be ignored if their names do not match any of the possible input variables.

* Year:		Year (YYYY)
* DOY: 		Day of the Year (0-366)
* Time: 	Decimal time (hrs), use the stdlong parameter to set the time zone
* Trad: 	Radiometric composite temperature (Celsius)
* Trad_0:	Radiometric composite temperature near sunrise (Celsius). OPTIONAL, only needed for DTD model.
* Tc:		Canopy component temperature (Celsius). OPTIONAL only needed for the TSEB-2T model
* Ts:		Soil component temperature (Celsius). OPTIONAL only needed for the TSEB-2T model
* VZA:		View Zenith Angle (Degrees)
* SZA:		Solar Zenith Angle (Degrees). OPTIONAL, will be estimated otherwise
* SAA:		Solar Azimuth Angle (Degrees). OPTIONAL, will be estimated otherwise
* Ta:		Air temperature (Celsius)
* Ta_0:		Air temperature near sunrise (Celsius). OPTIONAL only needed for DTD model
* u:		Wind speed (m/s)
* ea:		Vapor pressure (mb)
* p:		Atmospheric pressure (mb). OPTIONAL, will be estimated otherwise
* Sdn:		Incoming shortwave irradiance (W/m2)
* Ldn:		Incoming longwave irradiance (W/m2). OPTIONAL will be estimated otherwise
* LAI:		Effective Leaf Area Index (m2/m2)
* hc:		Canopy height (m)
* fc:		Fractional cover (0-1). OPTIONAL, will be set to full cover (fc=1) otherwise
* fg:		Green fraction (0-1). OPTIONAL, will be set to full green vegetation (fg=1) otherwise
* wc:		Canopy with to height ratio (m/m). OPTIONAL, will be set to spherical/squared canopies (wc=1) otherwise
* G:		Soil heat flux (W/m2). OPTIONAL, will be estimated otherwise

If any of those additional variables are not found in the table they will be internally estimated by TSEB or use default values. The order of the columns is not relevant, and neither whether there are additional columns in the table (they will be ignored if their names do not match any of the possible input variables.

## Configure the TSEB run
### Load or save a configuration file
A configuration file contains all the information required by pyTSEB to run without further user interaction, for instance by running *MAIN_TSEB_LocalImage.py* or *MAIN_TSEB_PointTimeSeries.py* scripts. An example of configuration file can be found [here](./Config_LocalImage.txt "Configuration file for processing an image") 

You can load one of these configuration files to fill all or some of the inputs required in the notebook GUI. Just press the button `Load Configuration File` and browse to the configuration file you want to load. All the valid information in the file will be appended to the GUI.

Likewise, after modifying or creating a pyTSEB configuration in the GUI, you can save it in a configuration file for future runs and for re-using without the need to manually add again redundant information. Press the button `Save Configuration File`, browse to the folder you want to store and either select an exisiting file (it will be overwritten) or type a new filename. A configuration file will be created and a message will appear in the notebook.

### Input/Output filenames
To add input files and/or create the output files the steps are the same for both GUIs. Press the `Browse ... ` button to load the open/save filedialog and browse the file. You can also directly type the path to the file in the text box.

In the case of the optional inputs for the image version, you can also type in the text box a number (use '.' as decimals) to use a constant value for the whole scene.

### pyTSEB parametrization
Finally, a set of tabs are displayed to configure additional parameters in pyTSEB, before running pyTSEB it is recommended to check along all the tabs to ensure that all the values are correct.

* TSEB Model: allows to choose which TSEB model is going to be run, click on the button with the model name you want... also remember that the input data has to be prepared for this type of model run, i.e. *Component Temperatures* needs as inputs canopy and soil temperature, see [Prepare the data](#Prepare-the-data)

* Site Description: introduce the site area coordinates (latitude and logitude in decimal degrees), the altitude above mean sea level (meters), the time zone used in the time information, and the height of measurement of air speed and temperature. Lat and lon are used to estimate solar angles together with observation time and time zone, if solar angles are missing in the input. The altitude to estimate barometric pressure if missing. Measurement heights are used for wind profile estimation and resistances to heat transport.

* Meteorology: only needed in the image GUI as meteo must be included as input for running a point time series. Day of the Year and decimal time are used to estimate solar angles as well as soil heat flux if the Santanello and Friedl model is used in G computation. If *DTD* is selected be aware to also include the air temperature near sunrise (time 0). LW irradiance and pressure are optional, you can leave either of these cells blank to let pyTSEB compute them.

* Spectral Properties. Set the bihemispherical reflectance and transmittance of individual leaves and the reflectance of the soil background. Set these properties for the PAR region (400-700nm) and for the NIR (700-2500nm). Also set the broadband hemispherical emissivity of leaves and soil.

* Canopy Description. Max alphaPT sets the a priori Priestley-Taylor parameter for potential canopy transpiration (default=1.26). LIDF param. set the Chi parameter of the Campbell's ellipsoidal leaf inclination distribution function (default Chi=1 for sphericall leaves) to estimate radiation transmission through the canopy. Leaf width set the characteristic size of the leaves used for estimating the canopy boundary resistance. Soil roughness sets the surface aerodynamic roughness for the bare soil, used in the estimation of the wind profile near the soil surface. Land cover type is used to estimate surface roughness: if CROPS or GRASS (lc=2 and 11) a simple ratio of the canopy height is used, if BROADLEAVED, CONIFER, or SHRUB (4,4 and 5 respectively), pyTSEB use the Dickinson and Schaudt & Dickinson (2000), based on LAI, fc, and hc.

* Additional options. Set how soil heat flux is estimated. If `Ratio of soil net radiation` is used a fixed ratio of the Rn_soil is applied (G=Gratio*Rn_soil, Gratio=0.35 by default). If `Constant or measured value` is selected the model will force G to be either the value typed in the corresponding cell (i.e. use 0 to ignore the soil heat flux in the energy balance) or, in the case of the point time series version, pyTSEB will use the values at the input table, in case a column named 'G' is present. If `Time dependent (Santanello & Friedl)` pyTSEB will estimate G as a sinusoidal time varying ratio of Rn_soil, with a maximum value corresponding to the Amplitude (default=0.35), a temporal shift from the peak of net radiation (+3h by default) and a shape (i.e. frequency, default=24h)

## TSEB outputs
Once TSEB is configured we will parse all the information in the widgets to run TSEB. A progress bar will show up and once done it will save the output files

### ProcessLocalImage.ipynb

* < Main Output File > whose name will be the one in the cell *Output File* will contain the bulk estimated fluxes with the following channels:
    1. Sensible heat flux (W m-2)
    2. Latent heat flux (W m-2)
    3. Net radiation (W m-2)
    4. Soil heat flux (W m-2)

* < Ancillary Output File > with the same name as the main input file with the suffix *_ancillary* added, will contain ancillary information from TSEB  with the following channels:
    1. Canopy sensible heat flux (W m-2)
    2. Canopy latent heat flux (W m-2)
    3. Evapotrasnpiration partitioning (canopy LE/total LE)
    4. Canopy temperature (K)
    5. Soil temperature (K)
    6. Net shortwave radiation (W m-2)
    7. Net longwave radiation (W m-2)
    8. Friction velocity (m s-1)
    9. Monin-Obukhov lenght (m)

### ProcessPointTimeSeries.ipynb
It will write an ASCII table in the output txt file with the following variables:
>Year, DOY, Time, LAI, f_g, skyl, VZA, SZA, SAA, Ldn, Rn_model, Rn_sw_veg, Rn_sw_soil, Rn_lw_veg, Rn_lw_soil, Tc, Ts, Tac, LE_model, H_model, LE_c, H_c, LE_s, H_s, flag, zo, d, G_model, R_s, R_x, R_a, u_friction, L, n_iterations

where *f_g* is the green fraction, *skyl* is the ratio of difuse radiation, *VZA*, *SZA* and *SAA* are view and solar angles, *Ldn* is downwelling longwave radiation, *Rn* is net radiation (*sw* and *lw* for shortwave and longwave, *veg* and *soil* for canopy and soil), *Tc* and *Ts* are canopy and soil temperatures, *LE* is latent heat flux , *H* is sensible heat flux, *flag* is a quality flag (255==BAD), *zo* and *d* are roughness leght and zero-plane displacement height, *G* is soil heat flux, *R_s*, *R_x* and *R_a* are resistances to heat and momentum transport, *u_friction* is friction velocity, *L* is the Monin-Obukhov lenght and *n_iterations* is the number of iterations in TSEB to achieve converge stability.

### Quality flags
pyTSEB might produce some *more* unrealible data that can be tracked with the quality flags:

* 0: Al Fluxes produced with no reduction of PT parameter (i.e. positive soil evaporation)
* 3: negative soil evaporation, forced to zero (the PT parameter is reduced in TSEB-PT and DTD)
* 5: No positive latent fluxes found, G recomputed to close the energy balance (G=Rn-H)
* 255: Arithmetic error. BAD data, it should be discarded

In addition for the component temperatures TSEB (TSEB-2T)
* 1: negative canopy latent heat flux, forced to zero
* 2: negative canopy sensible heat flux, forced to zero
* 4: negative soil sensible heat flux, forced to zero

## Display the results
Once TSEB is executed, we can also see a first glance of the results by plotting the bulk fluxes (latent heat flux, sensible heat flux, net radiation and soil heat flux)

The point time series version produces a time series plot with the capability to zoom, pan and query the actual flux values. The image version plots 4 syncronized pseudo-color images where you can also zoom and pan, displaying always the same are in the 4 plates. In addition you can save the figure in png format with the the save disk icon.