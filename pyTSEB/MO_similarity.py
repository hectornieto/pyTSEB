# This file is part of pyTSEB for processes related to the Monin-Obukhov Similarity Theory
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
Created on Apr 6 2015
@author: Hector Nieto (hector.nieto@ica.csic.es)

DESCRIPTION
===========
This package contains the main routines for estimating variables related to the
 Monin-Obukhov (MO) Similarity Theory, such as  MO length, adiabatic correctors
for heat and momentum transport. It requires the following package.

* :doc:`meteo_utils` for the estimation of meteorological variables.

PACKAGE CONTENTS
================
* :func:`calc_L` Monin-Obukhov length.
* :func:`calc_richardson` Richardson number.
* :func:`calc_u_star` Friction velocity.

Stability correction functions
------------------------------
* :func:`calc_Psi_H` Adiabatic correction factor for heat transport.
* :func:`calc_Psi_M` Adiabatic correction factor for momentum transport.
* :func:`CalcPhi_M_Brutsaert` [Brutsaert1992]_ similarity function for momentum transfer.
* :func:`CalcPhi_H_Dyer` [Dyer1974]_ similarity function for heat transfer.
* :func:`CalcPhi_M_Dyer` [Dyer1974]_ similarity function for momentum transfer.


"""

import numpy as np

from . import meteo_utils as met

# ==============================================================================
# List of constants used in MO similarity
# ==============================================================================
# von Karman's constant
KARMAN = 0.41
# acceleration of gravity (m s-2)
GRAVITY = 9.8

UNSTABLE_THRES = None
STABLE_THRES = None

def calc_L(ustar, T_A_K, rho, c_p, H, LE):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    T_A_K : float
        air temperature (Kelvin).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    H : float
        sensible heat flux (W m-2).
    LE : float
        latent heat flux (W m-2).

    Returns
    -------
    L : float
        Obukhov stability length (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.'''

    l_mo = calc_mo_length_hv(ustar, T_A_K, rho, c_p, H, LE)
    return np.asarray(l_mo)


def calc_mo_length(ustar, T_A_K, rho, c_p, H):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    T_A_K : float
        air temperature (Kelvin).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    H : float
        sensible heat flux (W m-2).
    LE : float
        latent heat flux (W m-2).

    Returns
    -------
    L : float
        Obukhov stability length (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.'''

    # Convert input scalars to numpy arrays
    ustar, T_A_K, rho, c_p, H = map(
        np.asarray, (ustar, T_A_K, rho, c_p, H))

    L = np.asarray(np.ones(ustar.shape) * float('inf'))
    i = H != 0
    L[i] = - c_p[i] * T_A_K[i] * rho[i] * ustar[i]**3 / (KARMAN * GRAVITY * H[i])
    return np.asarray(L)


def calc_mo_length_hv(ustar, T_A_K, rho, c_p, H, LE):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    T_A_K : float
        air temperature (Kelvin).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    H : float
        sensible heat flux (W m-2).
    LE : float
        latent heat flux (W m-2).

    Returns
    -------
    L : float
        Obukhov stability length (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.'''

    # Convert input scalars to numpy arrays
    ustar, T_A_K, rho, c_p, H, LE = map(
        np.asarray, (ustar, T_A_K, rho, c_p, H, LE))
    # first convert latent heat into rate of surface evaporation (kg m-2 s-1)
    Lambda = met.calc_lambda(T_A_K)  # in J kg-1
    E = LE / Lambda
    del LE, Lambda
    # Virtual sensible heat flux
    Hv = H + (0.61 * T_A_K * c_p * E)
    del H, E

    L = np.asarray(np.ones(ustar.shape) * float('inf'))
    i = Hv != 0
    L_const = np.asarray(KARMAN * GRAVITY / T_A_K)
    L[i] = -ustar[i]**3 / (L_const[i] * (Hv[i] / (rho[i] * c_p[i])))
    return np.asarray(L)

def calc_Psi_H(zoL):
    ''' Calculates the adiabatic correction factor for heat transport.

    Parameters
    ----------
    zoL : float
        stability coefficient (unitless).

    Returns
    -------
    Psi_H : float
        adiabatic corrector factor fof heat transport (unitless).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.
    '''
    # Avoid free convection situations
    if UNSTABLE_THRES is not None or STABLE_THRES is not None:
        zoL = np.clip(zoL, UNSTABLE_THRES, STABLE_THRES)
    Psi_H = psi_h_brutsaert(zoL)
    return np.asarray(Psi_H)

def psi_h_dyer(zol):

    gamma = 16
    beta = 5
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    psi_h = np.zeros(zol.shape)
    finite = np.isfinite(zol)
    # for stable and netural (zoL = 0 -> Psi_H = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    psi_h[i] = -beta * zol[i]
    # for unstable conditions
    i = np.logical_and(finite, zol < 0.0)
    x = (1 - gamma * zol[i])**0.25
    psi_h[i] = 2 * np.log((1 + x**2) / 2)
    return psi_h


def psi_h_brutsaert(zol):
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    psi_h = np.zeros(zol.shape)
    finite = np.isfinite(zol)

    # for stable and netural (zoL = 0 -> Psi_H = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    a = 6.1
    b = 2.5
    psi_h[i] = -a * np.log(zol[i] + (1.0 + zol[i]**b)**(1. / b))
    
    # for unstable conditions
    i = np.logical_and(finite, zol < 0.0)
    y = -zol[i]
    del zol
    c = 0.33
    d = 0.057
    n = 0.78
    psi_h[i] = ((1.0 - d) / n) * np.log((c + y**n) / c)
    return psi_h


def calc_Psi_M(zoL):
    ''' Adiabatic correction factor for momentum transport.

    Parameters
    ----------
    zoL : float
        stability coefficient (unitless).

    Returns
    -------
    Psi_M : float
        adiabatic corrector factor fof momentum transport (unitless).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.
    '''
    # Avoid free convection situations
    if UNSTABLE_THRES is not None or STABLE_THRES is not None:
        zoL = np.clip(zoL, UNSTABLE_THRES, STABLE_THRES)
    Psi_M = psi_m_brutsaert(zoL)
    return np.asarray(Psi_M)


def psi_m_dyer(zol):
    gamma = 16
    beta = 5
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    finite = np.isfinite(zol)
    psi_m = np.zeros(zol.shape)
    # for stable and netural (zoL = 0 -> Psi_M = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    psi_m[i] = -beta * zol[i]
    # for unstable conditions
    i = np.logical_and(finite, zol < 0.0)
    x = (1 - gamma * zol[i]) ** 0.25
    psi_m[i] = np.log((1 + x ** 2) / 2) + 2 * np.log((1 + x) / 2) \
               - 2 * np.arctan(x) + np.pi / 2.

    return psi_m


def psi_m_brutsaert(zol):
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    finite = np.isfinite(zol)
    psi_m = np.zeros(zol.shape)
    # for stable and netural (zoL = 0 -> Psi_M = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    a = 6.1
    b = 2.5
    psi_m[i] = -a * np.log(zol[i] + (1.0 + zol[i]**b)**(1.0 / b))
    # for unstable conditions
    i = np.logical_and(finite, zol < 0)
    y = -zol[i]
    del zol
    a = 0.33
    b = 0.41
    x = np.asarray((y / a)**0.333333)

    psi_0 = -np.log(a) + 3**0.5 * b * a**0.333333 * np.pi / 6.0
    y = np.minimum(y, b**-3)
    psi_m[i] = (np.log(a + y) - 3.0 * b * y**0.333333 +
                (b * a**0.333333) / 2.0 * np.log((1.0 + x)**2 / (1.0 - x + x**2)) +
                3.0**0.5 * b * a**0.333333 * np.arctan((2.0 * x - 1.0) / 3**0.5) +
                psi_0)

    return psi_m

def calc_richardson(u, z_u, d_0, T_R0, T_R1, T_A0, T_A1):
    '''Richardson number.

    Estimates the Bulk Richardson number for turbulence using
    time difference temperatures.

    Parameters
    ----------
    u : float
        Wind speed (m s-1).
    z_u : float
        Wind speed measurement height (m).
    d_0 : float
        Zero-plane displacement height (m).
    T_R0 : float
        radiometric surface temperature at time 0 (K).
    T_R1 : float
        radiometric surface temperature at time 1 (K).
    T_A0 : float
        air temperature at time 0 (K).
    T_A1 : float
        air temperature at time 1 (K).

    Returns
    -------
    Ri : float
        Richardson number.

    References
    ----------
    .. [Norman2000] Norman, J. M., W. P. Kustas, J. H. Prueger, and G. R. Diak (2000),
        Surface flux estimation using radiometric temperature: A dual-temperature-difference
        method to minimize measurement errors, Water Resour. Res., 36(8), 2263-2274,
        http://dx.doi.org/10.1029/2000WR900033.
    '''

    # See eq (2) from Louis 1979
    Ri = -(GRAVITY * (z_u - d_0) / T_A1) * \
          (((T_R1 - T_R0) - (T_A1 - T_A0)) / u**2) # equation (12) [Norman2000]
    return np.asarray(Ri)


def calc_u_star(u, z_u, L, d_0, z_0M):
    '''Friction velocity.

    Parameters
    ----------
    u : float
        wind speed above the surface (m s-1).
    z_u : float
        wind speed measurement height (m).
    L : float
        Monin Obukhov stability length (m).
    d_0 : float
        zero-plane displacement height (m).
    z_0M : float
        aerodynamic roughness length for momentum transport (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.
    '''

    # Covert input scalars to numpy arrays
    u, z_u, L, d_0, z_0M = map(np.asarray, (u, z_u, L, d_0, z_0M))

    # calculate correction factors in other conditions
    L[L == 0.0] = 1e-36
    Psi_M = calc_Psi_M((z_u - d_0) / L)
    Psi_M0 = calc_Psi_M(z_0M / L)
    del L
    u_star = u * KARMAN / (np.log((z_u - d_0) / z_0M) - Psi_M + Psi_M0)
    return np.asarray(u_star)
