# PyTSEB

## Synopsis

This project contains *Python* code for *Two Source Energy Balance* models (Priestley-Taylor **TSEB-PT**, 
Dual Time Difference **DTD** and TSEB with component soil and canopy temperatures **TSEB-2T**) 
for estimating sensible and latent heat flux (evapotranspiration) based on measurements of radiometric surface temperature. 

## :warning: Note to Users
We hope that our effort would enhance our interactions with other groups and let us learn about the situations in which the model produces poor or unrealistic results and causes the user(s) to conclude the model is not robust. In these situations, we would be very grateful to receive any feedback from your findings and, if possible, share the data and inputs used when implementing the model.  This could help us understand factors causing poor performance of the model which may be related to improperly prescribed inputs to the model or lead to  model improvements in its parameterizations by working together in collaboration.

For example, from previous experiences in which we assisted in configuring pyTSEB, we have noticed that the user(s) sometimes incorrectly prescribe some of the the inputs required by the model, causing the model to produce erroneous results which may lead to wrong conclusions regarding the robustness of the model. Examples of ill-prescribed model inputs include:  

1. parsing air temperature or wind speed measured at 2 or 10m for canopies taller than this height, which would violate Monin-Obukhov similarity theory
2. Wrong interpretation of the fractional cover parameter, which for TSEB is only needed for clumped/horizontally heterogeneous canopies, and thus the typical fractional cover variable retrieved with Earth Observation might not be appropriate for the landscape considered  as it usually represents the gap fraction for a horizontally homogeneous canopy
3. Inadequate definition of the green fraction (fg), which is not related to the (green) fractional cover but to the fraction of leaf area or plant area index  that is green.

On the other hand, there are likely conditions where pyTSEB performance will be less than satisfactory even with appropriate inputs, but it is these cases where we can determine the factors that cause the model to underperform and develop improvements/refinements benefitting future pyTSEB users and ET modellers.  Examples of how refinements of TSEB have been incorporated over different landscapes and environmental conditions are described in [Anderson et al. 2024]( https://doi.org/10.1016/j.agrformet.2024.109951).  Therefore, please do not hesitate in contacting us where you have faced issues related to pyTSEB  parametrizations or its behavior. We will be more than happy to collaborate with you and achieve a better understanding of model response to your application which will ultimately improve its applicability in  soil-plant-atmosphere water and energy  exchanges  in different environments.

To give us a better overview of the use of the model in different landscapes, climates and for different purposes, we would appreciate if you could cite its use in your publications using information of the [CITATION.cff](./CITATION.cff) file:

Nieto, H., Guzinski, R., & Kustas, W. P. (2018). pyTSEB: A python Two Source Energy Balance model for estimation of evapotranspiration with remote sensing data (Version 2.2) [Computer software]. https://doi.org/10.5281/zenodo.594732 

## Installation

Download the project to your local system, enter the download directory and then type

`pip install ./` 

if you want to install pyTSEB and its low-level modules in your Python distribution. 

The following Python libraries will be required:

- Numpy
- Pandas
- pyPro4Sail, at [https://github.com/hectornieto/pyPro4Sail]
- GDAL, for running TSEB over an image
- pandas
- netCDF4
- bokeh

With `conda`, you can create a complete environment with
```
conda env create -f environment.yml
```

## Code Example
### High-level example

The easiest way to get a feeling of TSEB and its configuration is through the provided ipython/jupyter notebooks. 
In a terminal shell, navigate to your working folder and type

- `jupyter notebook ProcessPointTimeSeries.ipynb` 
>for configuring and running TSEB over a time series of tabulated data

- `jupyter notebook ProcessLocalImage.ipynb` 
>for configuring and running TSEB over an image/scene using local meteorological data

In addition, you can also run TSEB with the scripts *TSEB_local_image_main.py* and *TSEB_point_time_series_main.py*, 
which will read an input configuration file (defaults are *Config_LocalImage.txt* and *Config_PointTimeSeries.txt* respectively). 
You can edit these configuration files or make a copy to fit your data and site characteristics and either run any of 
these two scripts in a Python GUI or in a terminal shell:

- `python TSEB_local_image_main.py <configuration file>`
> where \<configuration file> points to a customized configuration file... leave it blank if you want to use the default 
file *Config_LocalImage.txt*

- `python TSEB_point_time_series.py <configuration file>`
> where \<configuration file> points to a customized configuration file... leave it blank if you want to use the default 
file *Config_PointTimeSeries.txt*

### Low-level example
You can run any TSEB model or any related process in python by importing the module *TSEB* from the *pyTSEB* package. 
It will also import the ancillary modules (*resitances.py* as `res`, *netRadiation* as `rad`,
*MOsimilarity.py* as `MO`, *ClumpingIndex.py* as `CI` and *meteoUtils.py* as `met`)

```python
import pyTSEB.TSEB as TSEB 
output=TSEB.TSEB_PT(Tr_K, vza, Ta_K, u, ea, p, Sdn_dir, Sdn_dif, fvis, fnir, sza, Lsky, LAI, hc, emisVeg, emisGrd, spectraVeg, spectraGrd, z_0M, d_0, zu, zt)
```

You can type
`help(TSEB.TSEB_PT)`
to understand better the inputs needed and the outputs returned

The direct and difuse shortwave radiation (`Sdn_dir`, `Sdn_dif`, `fvis`, `fnir`) and the downwelling longwave radiation (`Lsky`) can be estimated by

```python
emisAtm = TSEB.rad.calc_emiss_atm(ea,Ta_K_1) # Estimate atmospheric emissivity from vapour pressure (mb) and air Temperature (K)
Lsky = emisAtm * TSEB.met.calc_stephan_boltzmann(Ta_K_1) # in W m-2
difvis,difnir, fvis,fnir=TSEB.rad.calc_difuse_ratio(Sdn,sza,press=p, Wv=1) # fraction of difuse and PAR/NIR radiation from shortwave irradiance (W m-2, solar zenith angle, atmospheric pressure and precipitable water vapour )
Skyl=difvis*fvis+difnir*fnir # broadband difuse fraction
Sdn_dir=Sdn*(1.0-Skyl)
Sdn_dif=Sdn*Skyl
```
   
## Basic Contents
The project consists of: 

1. lower-level modules with the basic functions needed in any resistance energy balance model 
2. higher-level scripts for easily running TSEB with tabulated data and/or satellite/airborne imagery.

### High-level modules
- *.pyTSEB/pyTSEB.py*, class object for TSEB scripting

- *ProcessPointTimeSeries.ipynb* and *ProcessLocalImage.ipynb* notebooks for using TSEB and configuring 
TSEB through a Graphical User Interface, GUI

- *TSEB_local_image_main.py* and *TSEB_point_time_series.py*, high level scripts for running TSEB 
through a configuration file (*Config_LocalImage.txt* or *Config_PointTimeSeries.txt*)

### Low-level modules
The low-level modules in this project are aimed at providing customisation and more flexibility in running TSEB. 
The following modules are included

- *.pyTSEB/TSEB.py*
> core functions for running different TSEB models (`TSEB_PT (*args,**kwargs)`, `TSEB_2T(*args,**kwargs)`, 
`DTD (*args,**kwargs)`), or a One Source Energy Balance model (`OSEB(*args,**kwargs)`). 

- *.pyTSEB/net_radiation.py*
> functions for estimating net radiation and its partitioning between soil and canopy

- *.pyTSEB/resistances.py*
> functions for estimating the different resistances for momemtum and heat transport and surface roughness

- *.pyTSEB/MO_similarity.py*
> functions for computing adiabatic corrections for heat and momentum transport, 
Monin-Obukhov length, friction velocity and wind profiles

- *.pyTSEB/clumping_index.py*
> functions for estimating the canopy clumping index and get effective values of Leaf Area Index

- *.pyTSEB/meteo_utils.py*
> functions for estimating meteorolgical-related variables such as density of air, 
heat capacity of air or latent heat of vaporization.

## API Reference
http://pytseb.readthedocs.org/en/latest/index.html

## Main Scientific References
- Norman,  J.  M.,  Kustas,  W.  P.,  Prueger,  J.  H.,  and  Diak,  G.  R.: Surface  flux  estimation  using  radiometric  temperature:  a  dual-temperature-difference method to minimize measurement errors, Water  Resour.  Res.,  36,  2263,  doi: 10.1029/2000WR900033, 2000
- Norman,  J.,  Kustas,  W.,  and  Humes,  K.:  A  two-source  approach for estimating soil and vegetation fluxes from observations of directional radiometric surface temperature, Agr. Forest Meteorol., 77, 263–293, doi: 10.1016/0168-1923(95)02265-Y, 1995
- Kustas, W. P. and Norman, J. M.: A two-source approach for estimating turbulent fluxes using multiple angle thermal infrared observations, Water Resour. Res., 33, 1495–1508, 199
- Kustas,  W.  P.  and  Norman,  J.  M.:  Evaluation  of  soil  and  vegetation heat flux prediction using a simple two-source model with radiometric  temperatures  for  partial  canopy  cover,  Agr.  Forest Meteorol., 94, 13–29, 199
- Guzinski, R., Nieto, H., Stisen, S., and Fensholt, R.: Inter-comparison of energy balance and hydrological models for land surface energy flux estimation over a whole river catchment, Hydrol. Earth Syst. Sci., 19, 2017-2036, doi:10.5194/hess-19-2017-2015, 2015.
- William P. Kustas, Hector Nieto, Laura Morillas, Martha C. Anderson, Joseph G. Alfieri, Lawrence E. Hipps, Luis Villagarcía, Francisco Domingo, Monica Garcia: Revisiting the paper “Using radiometric surface temperature for surface energy flux estimation in Mediterranean drylands from a two-source perspective”, Remote Sensing of Environment, In Press. doi:10.1016/j.rse.2016.07.024.


## Tests
The folder *./Input* contains examples for running TSEB in a tabulated time series (*ExampleTableInput.txt*) 
and in an image (*ExampleImage_\< variable >.tif*). Just run the high-level scripts with the configuration files 
provided by default and compare the resulting outputs with the files stored in *./Output/*

## Contributors
- **Hector Nieto** <hnieto@ias.csic.es> <hector.nieto.solana@gmail.com> main developer
- **Radoslaw Guzinski** main developer, tester
- **William P. Kustas** TSEB modeling, tester 
- **Ana Andreu** tester

## License
pyTSEB: a Python Two Source Energy Balance Model

Copyright 2016 Hector Nieto and contributors.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
