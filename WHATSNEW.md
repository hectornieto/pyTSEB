# What is new from pyTSEB v1.X
* Include 4SAIL canopy radiative transfer model for estimation of net radiation and retrieval of canopy and soil temperature using dual angle surface temperatures.
	- Guzinski, R., Nieto, H., Stisen, S., and Fensholt, R. (2015) Inter-comparison of energy balance and hydrological models for land surface energy flux estimation over a whole river catchment, Hydrol. Earth Syst. Sci., 19, 2017-2036, [http://dx/doi.org/10.5194/hess-19-2017-2015]. 

* Allow the modification of Kustas and Norman 1999 resistance parameter values (b,c and C').
	- William P. Kustas, Hector Nieto, Laura Morillas, Martha C. Anderson, Joseph G. Alfieri, Lawrence E. Hipps, Luis Villagarcía, Francisco Domingo, Monica Garcia, (In Press) Revisiting the paper “Using radiometric surface temperature for surface energy flux estimation in Mediterranean drylands from a two-source perspective”, Remote Sensing of Environment. [http://dx.doi.org/10.1016/j.rse.2016.07.024].

* Include additional resistance formulations based on Choudhury and Monteith (1988) and McNaughton and Van der Hurk (1995).
	- Choudhury, B. J., & Monteith, J. L. (1988). A four-layer model for the heat budget of homogeneous land surfaces. Quarterly Journal Royal Meteorological Society, 114(480), 373-398. [http://dx/doi.org/10.1002/qj.49711448006].
	- McNaughton, K. G., & Van den Hurk, B. J. J. M. (1995) A 'Lagrangian' revision of the resistors in the two-layer model for calculating the energy budget of a plant canopy. Boundary-Layer Meteorology, 74(3), 261-288. [http://dx/doi.org/10.1007/BF00712121].

* Include new One Source Energy Balance approaches for estimating fluxes for dense vegetation and bare soil under a sparse canopy.

* New code styling based on [PEP 8](https://www.python.org/dev/peps/pep-0008/) 

* Import modules based on package name.. e.g. `import pyTSEB.meteo_utils as met` 
