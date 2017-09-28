# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:50:52 2016

@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the main routines for estimating the wind profile above and within a canopy.
It requires the following package.

* :doc:`MO_similarity` for the estimation of adiabatic correctors.

Wind profile functions
----------------------
* :func:`calc_u_C` [Norman1995]_ canopy wind speed.
* :func:`calc_u_C_star` MOST canopy wind speed.
* :func:`calc_u_Goudriaan` [Goudriaan1977]_ wind speed profile below the canopy.
* :func:`calc_A_Goudriaan` [Goudriaan1977]_ wind attenuation coefficient below the canopy.
"""

import numpy as np

import pyTSEB.MO_similarity as MO  
#==============================================================================
# List of constants used in wind_profile
#==============================================================================

# Drag coefficient (Goudriaan 1977)
c_d = 0.2
# Relative turbulence intensity (Goudriaan 1977)
i_w = 0.5
# Von Karman constant
KARMAN = 0.4

def calc_u_C(u_friction, h_C, d_0, z_0M):
    '''[Norman1995]_ wind speed at the canopy, reformulated to use u_friction

    Parameters
    ----------
    u_friction : float
        Wind friction velocity (m s-1).
    h_C : float
        canopy height (m).
    d_0 : float
        zero-plane displacement height.
    z_0M : float
        aerodynamic roughness length for momentum transport (m).

    Returns
    -------
    u_C : float
        wind speed at the canop interface (m s-1).

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    # The original equation below has been refolmulated to use u_friction:
    # u_C = u * log((h_C - d_0) / z_0M)/(log ((z_u  - d_0) / z_0M)- Psi_M)
    u_C = np.log((h_C - d_0) / z_0M) * u_friction / KARMAN
    return np.asarray(u_C)


def calc_u_C_star(u_friction, h_C, d_0, z_0M, L=float('inf')):
    ''' MOST wind speed at the canopy

    Parameters
    ----------
    u_friction : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    d_0 : float
        zero-plane displacement height.
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    L : float, optional
        Monin-Obukhov length (m).

    Returns
    -------
    u_C : float
        wind speed at the canop interface (m s-1).
    '''

    Psi_M = MO.calc_Psi_M((h_C - d_0) / L)
    Psi_M0 = MO.calc_Psi_M(z_0M / L)

    # calcualte u_C, wind speed at the top of (or above) the canopy
    u_C = (u_friction * (np.log((h_C - d_0) / z_0M) - Psi_M + Psi_M0)) / KARMAN
    return np.asarray(u_C)


def calc_u_Goudriaan(u_C, h_C, LAI, leaf_width, z):
    '''Estimates the wind speed at a given height below the canopy.

    Parameters
    ----------
    U_C : float
        Windspeed at the canopy interface (m s-1).
    h_C : float
        canopy height (m).
    LAI : float
        Efective Leaf (Plant) Area Index.
    leaf_width : float
        effective leaf width size (m).
    z : float
        heigh at which the windsped will be estimated.

    Returns
    -------
    u_z : float
        wind speed at height z (m s-1).

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    .. [Goudriaan1977] Goudriaan (1977) Crop micrometeorology: a simulation study
 '''

    # extinction factor for wind speed
    a = calc_A_Goudriaan(h_C, LAI, leaf_width)
    u_z = u_C * np.exp(-a * (1.0 - (z / h_C)))  # Eq. 4.48 in Goudriaan 1977
    return np.asarray(u_z)


def calc_A_Goudriaan(h_C, LAI, leaf_width):
    ''' Estimates the extinction coefficient factor for wind speed

    Parameters
    ----------
    h_C : float
        canopy height (m)
    LAI : float
        Efective Leaf (Plant) Area Index
    leaf_width : float
        effective leaf width size (m)

    Returns
    -------
    a : float
        exctinction coefficient for wind speed through the canopy

    References
    ----------
    .. [Goudriaan1977] Goudriaan (1977) Crop micrometeorology: a simulation study
    '''

    # Equation in Norman et al. 1995
    k3_prime = 0.28
    a = k3_prime * LAI**(2. / 3.) * h_C**(1. / 3.) * leaf_width**(-1. / 3.)

    return np.asarray(a)
