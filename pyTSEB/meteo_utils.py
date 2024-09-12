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
@author: Hector Nieto (hector.nieto@ica.csic.es)


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
* :func:`calc_mixing_ratio` Ration of mass of water vapour to mass of dry air.
* :func:`calc_lapse_rate_moist` Moist-adiabatic lapse rate.
* :func:`flux_2_evaporation` Evaporation rate.
* :func:`bowen_ratio_closure` Corrects Energy Balance using presernvation of BR
'''

import numpy as np

# ==============================================================================
# List of constants used in Meteorological computations
# ==============================================================================
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
# acceleration of gravity (m s-2)
g = 9.8


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
        Latent heat of vaporisation (J kg-1).

    References
    ----------
    based on Eq. 3-1 Allen FAO98 '''

    Lambda = 1e6 * (2.501 - (2.361e-3 * (T_A_K - 273.15)))
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


def calc_psicr(c_p, p, Lambda):
    ''' Calculates the psicrometric constant.

    Parameters
    ----------
    c_p : float
        heat capacity of (moist) air at constant pressure (J kg-1 K-1).
    p : float
        atmopheric pressure (mb).
    Lambda : float
        latent heat of vaporzation (J kg-1).

    Returns
    -------
    psicr : float
        Psicrometric constant (mb C-1).'''

    psicr = c_p * p / (epsilon * Lambda)
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

def calc_rho_w(T_K):
    """
    density of air-free water ata pressure of 101.325kPa
    :param T_K:
    :return:
    density of water (kg m-3)
    """
    t = T_K - 273.15  # Temperature in Celsius
    rho_w = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
             - 46.170461e-6 * t**3 + 105.56302e-9 * t**4
             - 280.54253e-12 * t**5) / (1 + 16.897850e-3 * t)

    return rho_w

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
    """Calculates the Sun Zenith Angle (SZA).

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
    Adopted from Martha Anderson's fortran code for ALEXI which in turn was based on Cupid.
    """

    pid180 = np.pi / 180
    pid2 = np.pi / 2.0

    # Latitude computations
    xlat = np.radians(xlat)
    sinlat = np.sin(xlat)
    coslat = np.cos(xlat)

    # Declination computations
    kday = (year - 1977.0) * 365.0 + doy + 28123.0
    xm = np.radians(-1.0 + 0.9856 * kday)
    delnu = (2.0 * 0.01674 * np.sin(xm)
             + 1.25 * 0.01674 * 0.01674 * np.sin(2.0 * xm))
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
    """Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).

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
    """

    lat, lon, stdlon, doy, ftime = map(
        np.asarray, (lat, lon, stdlon, doy, ftime))

    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    EOT = (0.258 * np.cos(declination) - 7.416 * np.sin(declination)
           - 3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination))
    LC = (stdlon - lon) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = ftime - time_corr

    # Get the hour angle
    w = np.asarray((solar_time - 12.0) * 15.)

    # Get solar elevation angle
    sin_thetha = (np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(lat))
                  + np.sin(declination) * np.sin(np.radians(lat)))
    sun_elev = np.arcsin(sin_thetha)

    # Get solar zenith angle
    sza = np.pi / 2.0 - sun_elev
    sza = np.asarray(np.degrees(sza))

    # Get solar azimuth angle
    cos_phi = np.asarray(
        (np.sin(declination) * np.cos(np.radians(lat))
         - np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat)))
        / np.cos(sun_elev))
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360. - np.degrees(np.arccos(cos_phi[w > 0.0]))

    return np.asarray(sza), np.asarray(saa)


def calc_vapor_pressure(T_K):
    """Calculate the saturation water vapour pressure.

    Parameters
    ----------
    T_K : float
        temperature (K).

    Returns
    -------
    ea : float
        saturation water vapour pressure (mb).
    """

    T_C = T_K - 273.15
    ea = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return np.asarray(ea)


def calc_delta_vapor_pressure(T_K):
    """Calculate the slope of saturation water vapour pressure.

    Parameters
    ----------
    T_K : float
        temperature (K).

    Returns
    -------
    s : float
        slope of the saturation water vapour pressure (kPa K-1)
    """

    T_C = T_K - 273.15
    s = 4098.0 * (0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))) / ((T_C + 237.3)**2)
    return np.asarray(s)


def calc_mixing_ratio(ea, p):
    '''Calculate ratio of mass of water vapour to the mass of dry air (-)

    Parameters
    ----------
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).

    Returns
    -------
    r : float or numpy array
        mixing ratio (-)

    References
    ----------
    http://glossary.ametsoc.org/wiki/Mixing_ratio'''

    r = epsilon * ea / (p - ea)
    return r


def calc_lapse_rate_moist(T_A_K, ea, p):
    '''Calculate moist-adiabatic lapse rate (K/m)

    Parameters
    ----------
    T_A_K : float or numpy array
        air temperature at reference height (K).
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).

    Returns
    -------
    Gamma_w : float or numpy array
        moist-adiabatic lapse rate (K/m)

    References
    ----------
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate'''

    r = calc_mixing_ratio(ea, p)
    c_p = calc_c_p(p, ea)
    lambda_v = calc_lambda(T_A_K)
    Gamma_w = ((g * (R_d * T_A_K**2 + lambda_v * r * T_A_K)
               / (c_p * R_d * T_A_K**2 + lambda_v**2 * r * epsilon)))
    return Gamma_w


def flux_2_evaporation(flux, t_k=20 + 273.15, time_domain=1):
    '''Converts heat flux units (W m-2) to evaporation rates (mm time-1) to a given temporal window

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
    et : float or numpy array
        evaporation rate at the time_domain. Default mm h-1
    '''
    # Calculate latent heat of vaporization
    lambda_ = calc_lambda(t_k)  # J kg-1
    # Density of water
    rho_w = calc_rho_w(t_k)  # kg m-3
    et = flux / (rho_w * lambda_)  # m s-1
    # Convert instantaneous rate to the time_domain rate
    et = et * 1e3 * time_domain * 3600.  # mm
    return et


def bowen_ratio_closure(rn, g, h, le, br_range_exclude=[-1.3, -0.7]):
    br = h / le
    le_br = le.copy()
    h_br = h.copy()
    valid_br = np.logical_or(br <= br_range_exclude[0],
                             br >= br_range_exclude[1])

    le_br[valid_br] = (rn[valid_br] - g[valid_br]) / (1 + br[valid_br])
    h_br[valid_br] = rn[valid_br] - g[valid_br] - le_br[valid_br]
    return le_br, h_br

