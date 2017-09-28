# This file is part of pyTSEB for calculating the meteorolgical variables
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

'''
Created on Apr 6 2015
@author: Hector Nieto (hnieto@ias.csic.es)

Modified on feb 3 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains functions for estimating meteorological variables needed
in resistance energy balance models.

PACKAGE CONTENTS
================
* :func:`calc_c_p` Heat capacity of air at constant pressure.
* :func:`calc_lambda(T_A_K)` Latent heat of vaporization.
* :func:`calc_pressure` Barometric pressure.
* :func:`calc_psicr` Psicrometric constant.
* :func:`calc_rho` Density of air.
* :func:`calc_stephan_boltzmann` Stephan-Boltzmann law for blackbody radiation emission.
* :func:`calc_theta_s` Sun Zenith Angle.
* :func:`calc_sun_angles` Sun Zenith and Azimuth Angles.
* :func:`calc_vapor_pressure` Saturation water vapour pressure.
* :func:`calc_delta_vapor_pressure` Slope of saturation water vapour pressure.
'''

import numpy as np

#==============================================================================
# List of constants used in Meteorological computations
#==============================================================================
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# Psicrometric Constant kPa K-1
psicr = 0.0658
# gas constant for dry air, J/(kg*degK)
R_d = 287.04


def calc_c_p(p, ea):
    ''' Calculates the heat capacity of air at constant pressure.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).

    Returns
    -------
    c_p : heat capacity of (moist) air at constant pressure (J kg-1 K-1).

    References
    ----------
    based on equation (6.1) from Maarten Ambaum (2010):
    Thermal Physics of the Atmosphere (pp 109).'''
    
    # first calculate specific humidity, rearanged eq (5.22) from Maarten
    # Ambaum (2010), (pp 100)
    q = epsilon * ea / (p + (epsilon - 1.0) * ea)
    # then the heat capacity of (moist) air
    c_p = (1.0 - q) * c_pd + q * c_pv
    return np.asarray(c_p)


def calc_lambda(T_A_K):
    '''Calculates the latent heat of vaporization.

    Parameters
    ----------
    T_A_K : float
        Air temperature (Kelvin).

    Returns
    -------
    Lambda : float
        Latent heat of vaporisation (MJ kg-1).

    References
    ----------
    based on Eq. 3-1 Allen FAO98 '''

    Lambda = 2.501 - (2.361e-3 * (T_A_K - 273.15))
    return np.asarray(Lambda)


def calc_pressure(z):
    ''' Calculates the barometric pressure above sea level.

    Parameters
    ----------
    z: float
        height above sea level (m).

    Returns
    -------
    p: float
        air pressure (mb).'''

    p = 1013.25 * (1.0 - 2.225577e-5 * z)**5.25588
    return np.asarray(p)


def calc_psicr(p, Lambda):
    ''' Calculates the psicrometric constant.

    Parameters
    ----------
    p : float
        atmopheric pressure (mb).
    Lambda : float
        latent heat of vaporzation (MJ kg-1).

    Returns
    -------
    psicr : float
        Psicrometric constant (mb C-1).'''

    psicr = 0.00163 * p / (Lambda)
    return np.asarray(psicr)


def calc_rho(p, ea, T_A_K):
    '''Calculates the density of air.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).
    T_A_K : float
        air temperature at reference height (Kelvin).

    Returns
    -------
    rho : float
        density of air (kg m-3).

    References
    ----------
    based on equation (2.6) from Brutsaert (2005): Hydrology - An Introduction (pp 25).'''

    # p is multiplied by 100 to convert from mb to Pascals
    rho = ((p * 100.0) / (R_d * T_A_K)) * (1.0 - (1.0 - epsilon) * ea / p)
    return np.asarray(rho)


def calc_stephan_boltzmann(T_K):
    '''Calculates the total energy radiated by a blackbody.

    Parameters
    ----------
    T_K : float
        body temperature (Kelvin)

    Returns
    -------
    M : float
        Emitted radiance (W m-2)'''

    M = sb * T_K**4
    return np.asarray(M)


def calc_theta_s(xlat, xlong, stdlng, doy, year, ftime):
    '''Calculates the Sun Zenith Angle (SZA).

    Parameters
    ----------
    xlat : float
        latitude of the site (degrees).
    xlong : float
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    year : float
        year of measurement .
    ftime : float
        time of measurement (decimal hours).

    Returns
    -------
    theta_s : float
        Sun Zenith Angle (degrees).

    References
    ----------
    Adpopted from Martha Anderson's fortran code for ALEXI which in turn was based on Cupid.'''

    pid180 = np.pi / 180
    pid2 = np.pi / 2.0
    # Latitude computations
    xlat = np.radians(xlat)
    sinlat = np.sin(xlat)
    coslat = np.cos(xlat)
    # Declination computations
    kday = (year - 1977.0) * 365.0 + doy + 28123.0
    xm = np.radians(-1.0 + 0.9856 * kday)
    delnu = 2.0 * 0.01674 * np.sin(xm) + \
        1.25 * 0.01674 * 0.01674 * np.sin(2.0 * xm)
    slong = np.radians((-79.8280 + 0.9856479 * kday)) + delnu
    decmax = np.sin(np.radians(23.44))
    decl = np.arcsin(decmax * np.sin(slong))
    sindec = np.sin(decl)
    cosdec = np.cos(decl)
    eqtm = 9.4564 * np.sin(2.0 * slong) / cosdec - 4.0 * delnu / pid180
    eqtm = eqtm / 60.0
    # Get sun zenith angle
    timsun = ftime  # MODIS time is already solar time
    hrang = (timsun - 12.0) * pid2 / 6.0
    theta_s = np.arccos(sinlat * sindec + coslat * cosdec * np.cos(hrang))
    # if the sun is below the horizon just set it slightly above horizon
    theta_s = np.minimum(theta_s, pid2 - 0.0000001)
    theta_s = np.degrees(theta_s)
    return np.asarray(theta_s)


def calc_sun_angles(lat, lon, stdlon, doy, ftime):
    '''Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).

    Parameters
    ----------
    lat : float
        latitude of the site (degrees).
    long : float
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    ftime : float
        time of measurement (decimal hours).

    Returns
    -------
    sza : float
        Sun Zenith Angle (degrees).
    saa : float
        Sun Azimuth Angle (degrees).

    '''

    lat, lon, stdlon, doy, ftime = map(
        np.asarray, (lat, lon, stdlon, doy, ftime))
    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    EOT = 0.258 * np.cos(declination) - 7.416 * np.sin(declination) - \
        3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination)
    LC = (stdlon - lon) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = ftime - time_corr
    # Get the hour angle
    w = np.asarray((solar_time - 12.0) * 15.)
    # Get solar elevation angle
    sin_thetha = np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(lat)) + \
         np.sin(declination) * np.sin(np.radians(lat))
    sun_elev = np.arcsin(sin_thetha)
    # Get solar zenith angle
    sza = np.pi / 2.0 - sun_elev
    sza = np.asarray(np.degrees(sza))
    # Get solar azimuth angle
    cos_phi = np.asarray(
        (np.sin(declination) * np.cos(np.radians(lat)) -
         np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat))) /
        np.cos(sun_elev))
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360. - np.degrees(np.arccos(cos_phi[w > 0.0]))
    return np.asarray(sza), np.asarray(saa)


def calc_vapor_pressure(T_K):
    '''Calculate the saturation water vapour pressure.

    Parameters
    ----------
    T_K : float
        temperature (K).

    Returns
    -------
    ea : float
        saturation water vapour pressure (mb).'''

    T_C = T_K - 273.15
    ea = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return np.asarray(ea)


def calc_delta_vapor_pressure(T_K):
    '''Calculate the slope of saturation water vapour pressure.

    Parameters
    ----------
    T_K : float
        temperature (K).

    Returns
    -------
    s : float
        slope of the saturation water vapour pressure (kPa K-1)'''

    T_C = T_K - 273.15
    s = 4098.0 * (0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))) / \
        ((T_C + 237.3)**2)
    return np.asarray(s)

def flux_2_evaporation(flux,T_K=20+273.15,time_domain=1):
    '''Converts heat flux units (W m-2)to evaporation rates (mm time-1) to a given temporal window

    Parameters
    ----------
    flux : float or numpy array
        heat flux value to be converted, 
        usually refers to latent heat flux LE to be converted to ET
    T_K : float or numpy array
        environmental temperature in Kelvin. Default=20 Celsius
    time_domain : float
        Temporal window in hours. Default 1 hour (mm h-1)
    
    Returns
    -------
    ET : float or numpy array
        evaporation rate at the time_domain. Default mm h-1
    '''
    # Calculate latent heat of vaporization
    lambda_=calc_lambda(T_K)*1e6 #J kg-1
    ET=flux/lambda_ # kg s-1
    # Convert instantaneous rate to the time_domain rate
    ET=ET*time_domain*3600.
    return ET
    
