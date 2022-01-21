# Use of the Notebooks' GUI for running pyTSEB

## Prepare the data

### ProcessLocalImage.ipynb
#### Main input image
The main input data in this case is an image containing the radiometric temperature(s). Any [GDAL compatible raster format](http://www.gdal.org/formats_list.html) should work except those based on datasets (i.e. HDF, netCDF, etc.). The input temperature data should be in Kelvin. Depending on the TSEB model to use you have to be sure to include in the raster the following bands (in the given order):

* Priestley-Taylor TSEB (**TSEB-PT**)
	1. Radiometric surface temperature

* Dual-Time Difference TSEB (**DTD**)
	1. Radiometric surface temperature around noon
	2. Radiometric surface temperature around sunrise

* Composite temperatures TSEB( **TSEB-2T**)
	1. Canopy temperature
	2. Soil temperature

#### Ancillary input
The following variables can be provided either in an image format, in which case their size should exactly match the size of the temperature images, or as a single constant value to use in the whole scene

* Effective Leaf Area Index
* View Zenith Angle (degrees)
* Vegetation fractional cover, fc (0-1)
* Canopy height (m)
* Canopy width-to-height ratio (w/hc)
* Vegetation green fraction, fg (0-1)

The Processing mask is also optional, and if provided only the pixels with a mask value >0 will be processed.

### ProcessPointTimeSeries.ipynb
All the input information in the point time series versions must be provided in an ASCI (text) file containing a tab ("\t") delimited table. The first row of the table should contain column headers and at least the following field names should be included (case sensitive):

>* **TSEB-PT** : `Year, DOY, Time, Trad, VZA, Ta, u, ea, Sdn, LAI & hc`
>* **DTD** : `Year, DOY, Time, Trad_0, Trad, VZA, Ta_0, Ta, u, ea, Sdn, LAI & hc`
>* **TSEB-2T** : `Year, DOY, Time, Tc, Ts, Ta, u, ea, Sdn, LAI & hc`

The order of the columns is not relevant, and neither whether there are additional columns in the table (they will be ignored if their names do not match any of the possible input variables).

* Year:		Year (YYYY)
* DOY: 		Day of the Year (0-366)
* Time: 	Decimal local-solar time (hrs), use the stdlong parameter to set the time zone
* Trad: 	Radiometric composite temperature (Kelvin)
* Trad_0:	Radiometric composite temperature near sunrise (Kelvin). OPTIONAL, only needed for DTD model.
* Tc:		Canopy component temperature (Kelvin). OPTIONAL only needed for the TSEB-2T model
* Ts:		Soil component temperature (Kelvin). OPTIONAL only needed for the TSEB-2T model
* VZA:		View Zenith Angle (Degrees)
* SZA:		Solar Zenith Angle (Degrees). OPTIONAL, will be estimated otherwise
* SAA:		Solar Azimuth Angle (Degrees). OPTIONAL, will be estimated otherwise
* Ta:		Air temperature (Kelvin)
* Ta_0:		Air temperature near sunrise (Kelvin). OPTIONAL only needed for DTD model
* u:		Wind speed (m/s)
* ea:		Vapor pressure (mb)
* p:		Atmospheric pressure (mb). OPTIONAL, will be estimated otherwise
* Sdn:		Incoming shortwave irradiance (W/m2)
* Ldn:		Incoming longwave irradiance (W/m2). OPTIONAL will be estimated otherwise
* LAI:		Effective Leaf Area Index (m2/m2)
* hc:		Canopy height (m)
* fc:		Vegetation fractional cover (0-1). OPTIONAL, will be set to full cover (fc=1) otherwise
* fg:		Vegetation green fraction (0-1). OPTIONAL, will be set to full green vegetation (fg=1) otherwise
* wc:		Canopy with to height ratio (m/m). OPTIONAL, will be set to spherical/squared canopies (wc=1) otherwise
* G:		Soil heat flux (W/m2). OPTIONAL, will be estimated otherwise

If any of those additional variables are not found in the table they will be internally estimated by TSEB or default values will be used. 

## Configure the TSEB run
### Load or save a configuration file
A configuration file contains all the information required by pyTSEB to run without further user interaction, for instance by running *MAIN_TSEB_LocalImage.py* or *MAIN_TSEB_PointTimeSeries.py* scripts. An example of configuration file can be found [here]( https://github.com/hectornieto/pyTSEB/blob/master/Config_LocalImage.txt "Configuration file for processing an image") 

You can load one of these configuration files to fill all or some of the inputs required in the notebook GUI. Just press the button `Load Configuration File` and browse to the configuration file you want to load. All the valid information in the file will be displayed in the GUI.

Likewise, after modifying or creating a pyTSEB configuration in the GUI, you can save it in a configuration file for future runs and for re-using without the need to manually add again redundant information. Press the button `Save Configuration File`, browse to the folder you want to store the configuration and either select an existing file (it will be overwritten) or type a new filename. A configuration file will be created and a message will appear in the notebook.

### Input/Output filenames
To add input files and/or create the output files the steps are the same for both GUIs. Press the `Browse ... ` button to load the open/save file dialog and browse to the file. You can also directly type the path to the file in the text box.

In the case of the optional inputs for the image version, you can also type in the text box a number (use '.' as decimals) to use a constant value for the whole scene.

### pyTSEB parametrization
Finally, a set of tabs are displayed to configure additional parameters in pyTSEB. Before running pyTSEB it is recommended to check through all the tabs to ensure that all the values are correct.

* TSEB Model: to choose which TSEB model is going to be run, click on the button with the model name you want. Also remember that the input data has to be prepared for this type of model run, e.g. *TSEB-2T* needs as inputs canopy and soil temperature, see [Prepare the data](#prepare-the-data)

* Site Description: introduce the site area coordinates (latitude and logitude in decimal degrees), the altitude above mean sea level (meters), the time zone used in the time information, and the height of measurement of wind speed and air temperature. Lat and lon are used to estimate solar angles together with observation time and time zone, if solar angles are missing in the input. The altitude is used to estimate barometric pressure if missing. Measurement heights are used for wind profile estimation and to calculate resistances to heat transport.

* Meteorology: only needed in the image GUI as meteo must be included as input for running a point time series. Day of the Year and decimal time are used to estimate solar angles as well as soil heat flux if the Santanello and Friedl model is used in G computation. If *DTD* is selected be aware to also include the air temperature close to sunrise (time 0). LW irradiance and pressure are optional, you can leave either of these cells blank to let pyTSEB compute them.

* Spectral Properties: set the bihemispherical reflectance and transmittance of individual leaves and the reflectance of the soil background. Set these properties for the PAR region (400-700nm) and for the NIR (700-2500nm). Also set the broadband hemispherical emissivity of leaves and soil.

* Canopy Description: max alphaPT sets the a priori Priestley-Taylor parameter for potential canopy transpiration (default=1.26). LIDF param. sets the Chi parameter of the Campbell's ellipsoidal leaf inclination distribution function (default Chi=1 for spherical leaves) used to estimate radiation transmission through the canopy. Leaf width sets the characteristic size of the leaves used for estimating the canopy boundary resistance. Soil roughness sets the surface aerodynamic roughness for the bare soil, used in the estimation of the wind profile near the soil surface. Land cover type is used to estimate surface roughness: if CROPS or GRASS (lc=2 and 11) a simple ratio of the canopy height is used, if BROADLEAVED, CONIFER, or SHRUB (3,4 and 5 respectively), pyTSEB uses the Schaudt & Dickinson (2000) method, based on LAI, fc, and hc.
Check `Canopy in Rows` to calculate radiation partitioning in row crops. In that case you also have to provide the predominant row crop direction (degrees) and the height of the first living branch as a ratio of the total canopy height (hb=0 if canopy starts at the soil level).

* Resistance model: select which model to use in estimating the canopy boundary and soil resistances to heat and momentum transport. If Kustas and Norman 1999 model is selected you can also change the empirical coefficients used in Rx and Rs.

* Additional options: set how soil heat flux is estimated. If `Ratio of soil net radiation` is used then a fixed ratio of the Rn_soil is applied (G=Gratio*Rn_soil, Gratio=0.35 by default). If `Constant or measured value` is selected the model will force G to be either the value typed in the corresponding cell (i.e. use 0 to ignore the soil heat flux in the energy balance) or, in the case of the point time series version, pyTSEB will use the values at the input table, in case a column named 'G' is present. If `Time dependent (Santanello & Friedl)` pyTSEB will estimate G as a sinusoidal time varying ratio of Rn_soil, with a maximum value corresponding to the Amplitude (default=0.35), a temporal shift from the peak of net radiation (+3h by default) and a shape (i.e. frequency, default=24h).

## TSEB outputs
Once TSEB is configured we will parse all the information in the widgets and run the selected model. A progress bar will show up and once the processing is done the output files will be saved.

### ProcessLocalImage.ipynb

* < Main Output File > whose name is specified in the cell *Output File*, will contain the bulk estimated fluxes with the following channels:
    1. Net radiation (W m-2)
    2. Sensible heat flux (W m-2)
    3. Latent heat flux (W m-2)
    4. Soil heat flux (W m-2)

* < Ancillary Output File > with the same name as the main input file but with a suffix *_ancillary* added, will contain ancillary information from TSEB  with the following channels:
    1. Net shortwave radiation (W m-2)
    2. Net longwave radiation (W m-2)
    3. Net radiation divergence at the canopy (W m-2)
    4. Canopy sensible heat flux (W m-2)
    5. Canopy latent heat flux (W m-2)
    6. Evapotrasnpiration partitioning (canopy LE/total LE)
    7. Canopy temperature (K)
    8. Soil temperature (K)
    9. Aerodynamic resistance (s m-1)
    10. Bulk canopy resistance to heat transport (s m-1)
    11. Soil resistance to heat transport (s m-1)
    12. Monin-Obukhov lenght (m)
    13. Friction velocity (m s-1)
    14. Quality Flag (unitless)

### ProcessPointTimeSeries.ipynb
An ASCII table with the following variables will be written in the output text file::
>Year, DOY, Time, LAI, f_g, skyl, VZA, SZA, SAA, Ldn, Rn_model, Rn_sw_veg, Rn_sw_soil, Rn_lw_veg, Rn_lw_soil, Tc, Ts, Tac, LE_model, H_model, LE_c, H_c, LE_s, H_s, flag, zo, d, G_model, R_s, R_x, R_a, u_friction, L, n_iterations

where variables with the same name as input variables have the same meaning, and for the others: *skyl* is the ratio of diffuse radiation, *Rn_model* is net radiation (W m-2; *sw* and *lw* subscripts stand for shortwave and longwave, *veg* and *soil* for canopy and soil), *Tc*, *Ts* and *Tac* are canopy, soil and inter-canopy-air temperatures (K), *LE_model* is latent heat flux and *H_model* is sensible heat flux (W m-2; subscripts *c* and *s* stand for canopy and soil respectively), *flag* is a quality flag (255==BAD), *zo* and *d* are roughness length and zero-plane displacement height (m), *R_s*, *R_x* and *R_a* are resistances to heat and momentum transport (s m-1), *u_friction* is friction velocity (m s-1), *L* is the Monin-Obukhov length (m) and *n_iterations* is the number of iterations of TSEB needed to achieve model convergence.

### Quality flags
pyTSEB might produce some *more* unreliable data that can be tracked with the quality flags:

* 0: Al Fluxes produced with no reduction of PT parameter (i.e. positive soil evaporation)
* 3: negative soil evaporation, forced to zero (the PT parameter is reduced in TSEB-PT and DTD)
* 5: No positive latent fluxes found, G recomputed to close the energy balance (G=Rn-H)
* 255: Arithmetic error. BAD data, it should be discarded

In addition for the component temperatures TSEB (TSEB-2T):

* 1: negative canopy latent heat flux, forced to zero
* 2: negative canopy sensible heat flux, forced to zero
* 4: negative soil sensible heat flux, forced to zero

## Display the results
Once TSEB is executed, we can also have a first glance of the results by plotting the bulk fluxes (latent heat flux, sensible heat flux, net radiation and soil heat flux).

The point time series version produces a time series plot with the capability to zoom, pan and query the actual flux values. The image version plots 4 synchronized pseudo-colour images where you can also zoom and pan, displaying always the same area in the 4 plates. In addition you can save the figure in png format with the save disk icon.
