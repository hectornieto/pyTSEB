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
@author: Hector Nieto (hnieto@ias.csic.es)

Modified on Jan 27 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the main routines for estimating variables related to the Monin-Obukhov (MO) Similarity Theory,
such as  MO length, adiabatic correctors for heat and momentum transport and wind profile through a canopy. It
requires the following package.

* :doc:`meteoUtils` for the estimation of meteorological variables.

PACKAGE CONTENTS
================
* :func:`CalcL` Monin-Obukhov length.
* :func:`CalcRichardson` Richardson number.
* :func:`CalcU_star` Friction velocity.

Stability correction functions
------------------------------
* :func:`CalcPsi_H` Adiabatic correction factor for heat transport.
* :func:`CalcPsi_M` Adiabatic correction factor for momentum transport.
* :func:`CalcPsi_M_B92` Adiabatic correction factor for momentum transport [Brutsaert1992]_.

Wind profile functions
----------------------
* :func:`CalcU_C` [Norman1995]_ canopy wind speed.
* :func:`CalcU_Goudriaan` [Goudriaan1977]_ wind speed profile below the canopy.
* :func:`CalcA_Goudriaan` [Goudriaan1977]_ wind attenuation coefficient below the canopy.

"""

#==============================================================================
# List of constants used in MO similarity   
#==============================================================================
# von Karman's constant
k=0.4 
#acceleration of gravity (m s-2)
gravity=9.8
# Drag coefficient (Goudriaan 1977)
c_d=0.2
# Relative turbulence intensity (Goudriaan 1977)
i_w=0.5

import meteoUtils as met

def CalcL (ustar, Ta_K, rho, c_p, H, LE):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    Ta_K : float
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

    # first convert latent heat into rate of surface evaporation (kg m-2 s-1)
    Lambda = met.CalcLambda(Ta_K)*1e6 #in J kg-1
    E = LE / Lambda
    # Virtual sensible heat flux
    Hv=H+(0.61*Ta_K*c_p*E)
    if Hv!=0:
        L_const = k*gravity/Ta_K
        L = -ustar**3 / ( L_const*(Hv/(rho*c_p) ))
    else:
        L = float('inf')
    return L
    
def CalcPsi_H (zoL):
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
    
    from math import log
    Psi_H = 0.0
    #for stable and netural (zoL = 0 -> Psi_H = 0) conditions
    if zoL>=0.0:
        a = 6.1
        b = 2.5
        Psi_H = -a * log( zoL + (1.0 + zoL**b)**(1./b))
    # for unstable conditions
    else:
        y = -zoL
        c = 0.33		
        d = 0.057
        n = 0.78
        Psi_H = ((1.0-d)/n) * log((c + y**n)/c)
    return Psi_H

def CalcPsi_M (zoL):
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

    from math import log, pi,atan
    Psi_M = 0.0
    # for stable and netural (zoL = 0 -> Psi_M = 0) conditions
    if zoL>=0.0:
        a = 6.1 
        b = 2.5
        Psi_M = -a * log( zoL + (1.0 + zoL**b)**(1.0/b))
    # for unstable conditions
    else:
        y = -zoL
        a = 0.33
        b = 0.41
        x = (y/a)**0.333333
        Psi_0 = -log(a) + 3**0.5*b*a**0.333333*pi/6.0
        y = min(y, b**-3)
        Psi_M = (log(a + y) - 3.0*b*y**0.333333 + (b*a**0.333333)/2.0 * log((1.0+x)**2/(1.0-x+x**2))+
            3.0**0.5*b*a**0.333333*atan((2.0*x-1.0)/3**0.5) + Psi_0)
    return Psi_M

def CalcPsi_M_B92 (zoL):
    ''' Adiabatic correction factor for momentum transport [Brutsaert1992]_
    
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
    .. [Brutsaert1992] Brutsaert, W. (1992). Stability correction functions for the mean wind speed and
        temperature in the unstable surface layer. Geophysical research letters, 19(5), 469-472,
        http://dx.doi.org/10.1029/92GL00084.
    '''

    from math import log, pi
    Psi_M = 0.0
    # for stable and netural (zoL = 0 -> Psi_M = 0) conditions, Eq. 2.59 in Brutasert 2005  
    if zoL>=0.0:
        a = 6.1 
        b = 2.5
        Psi_M = -a * log( zoL + (1.0 + zoL**b)**(1.0/b))
    # for unstable conditions
    else:
        y=-zoL
        if y <0.0059:
            Psi_M=0.0
        elif y<=15.025:
            y_c=0.0059
            Psi_M = 1.47*log((0.28+y**0.75)/(0.28+(0.0059+y_c)**0.75))-1.29*(y**(1./3.)-(0.0059+y_c)**(1./3.))
        else:
            y_c=0.0059
            y=15.025
            Psi_M = 1.47*log((0.28+y**0.75)/(0.28+(0.0059+y_c)**0.75))-1.29*(y**(1./3.)-(0.0059+y_c)**(1./3.))

            
    return Psi_M

def CalcRichardson (u, z_u, d_0, T_R0, T_R1, T_A0, T_A1):
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
    Ri = -(gravity * (z_u - d_0) / T_A1) * (((T_R1 - T_R0) - (T_A1 - T_A0)) / u**2)
    return Ri

def CalcU_C (u, h_C, d_0, z_0M, z_u,L):
    '''[Norman1995]_ wind speed at the canopy
    
    Parameters
    ----------
    u : float
        Wind speed measured at heigth z_u (m s-1).
    h_C : float
        canopy height (m).
    d_0 : float
        zero-plane displacement height.
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    z_u:  float
        Height of measurement of wind speeed.
    L : float
     Monin Obukhov length.
    
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

    from math import log
    Psi_M= CalcPsi_M((h_C - d_0)/L)
    # calcualte u_C, wind speed at the top of (or above) the canopy
    u_C = u*log ((h_C  - d_0) / z_0M)/(log ((z_u  - d_0) / z_0M)- Psi_M)
    return u_C

def CalcU_Goudriaan (u_C, h_C, LAI, leaf_width, z):
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

    from math import exp
    a=CalcA_Goudriaan (h_C,LAI,leaf_width) # extinction factor for wind speed
    u_z = u_C * exp(-a * (1.0 - (z/h_C))) # Eq. 4.48 in Goudriaan 1977
    return u_z

def CalcA_Goudriaan (h_C,LAI,leaf_width):
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
    k3_prime=0.28
    a = k3_prime * LAI**(2./3.) * h_C**(1./3.) * leaf_width**(-1./3.)

    return a
    
def CalcU_star (u, z_u, L, d_0,z_0M, useRi=False):
    '''Friction velocity.
    
    Parameters
    ----------
    u : float
        wind speed above the surface (m s-1).
    z_u : float
        wind speed measurement height (m).
    L : float
        Obukhov stability length (m) or Richardson number, see useRi variable.
    d_0 : float
        zero-plane displacement height (m).
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    useRi : bool, optional 
        Use the Richardson number istead of MO length for adiabatic correction factors.
 
    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.
    ''' 
    from math import log
    Psi_M=0.0
    Psi_M0=0.0
    if useRi: 
        #use the approximation Ri ~ (z-d_0)./L from end of section 2.	2 from Norman et. al., 2000 (DTD paper) 
        Psi_M = CalcPsi_M(L);
        Psi_M0 = CalcPsi_M(L/(z_u - d_0)*z_0M)
    else:
        #calculate correction factors in other conditions
        if L == 0.0: L=1e-36
        Psi_M= CalcPsi_M((z_u - d_0)/L)
        Psi_M0 = CalcPsi_M(z_0M/L)
    u_star = u * k / ( log( (z_u - d_0) / z_0M ) - Psi_M + Psi_M0)
    return u_star
