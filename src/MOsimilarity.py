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

Modified on Dec 30 2015
@author: Hector Nieto (hnieto@ias.csic.es)

Routines for estimating variables related to the Monin-Obukhov (MO) Similarity Theory,
such as  MO lenght and adiabatic correctors for heat and momentum transport
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
    '''Calculates the Monin-Obukhov lenght

    Parameters
    ----------
    ustar : friction velocity (m s-1)
    Ta_K : air temperature (Kelvin)
    rho : air density (kg m-3)
    c_p : Heat capacity of air at constant pressure (J kg-1 K-1)
    H : sensible heat flux (W m-2)
    LE : latent heat flux (W m-2)
    
    Returns
    -------
    L : Obukhov stability length (m)

    based on equation (2.46) from Brutsaert (2005): 
    Hydrology - An Introduction (pp 46)'''

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
    ''' Calculates the adiabatic correction factor for heat transport
    
    Parameter
    ---------
    zoL : stability coefficient (unitless)
    
    Returns
    -------
    Psi_H : adiabatic corrector factor fof heat transport (unitless)
    
    based on equations (2.59) for stable and (2.64) for unstable of Brutsaert, 2005 
    "Hydrology, an introduction" book'''
    
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
    ''' Calculates the adiabatic correction factor for momentum transport
    
    Parameter
    ---------
    zoL : stability coefficient (unitless)
    
    Returns
    -------
    Psi_M : adiabatic corrector factor fof momentum transport (unitless)
    
    based on equations (2.59) for stable and (2.63) for unstable of Brutsaert, 2005 
    "Hydrology, an introduction" book'''

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
    ''' Calculates the adiabatic correction factor for momentum transport
    
    Parameter
    ---------
    zoL : stability coefficient (unitless)
    
    Returns
    -------
    Psi_M : adiabatic corrector factor fof momentum transport (unitless)
    
    based on equation (15) for unstable conditions of Brutsaert, 1992 
    "Stability correction fucntions for the mean wind speed and temperatures 
    in the unstable surface layer. Geophysical Research Letters, 19 (5), 479-472'''

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
    '''Estimates the Bulk Richardson number for turbulence using 
        time difference temperatures
    
    Parameters
    ----------
    u : Wind speed (m s-1)
    z_u : Wind speed measurement height (m)
    d_0 : Zero-plane displacement height (m)
    T_R0 : radiometric surface temperature at time 0 (K)
    T_R1 : radiometric surface temperature at time 1 (K)
    T_A0 : air temperature at time 0 (K)
    T_A1 : air temperature at time 1 (K)
    
    Returns
    -------
    Ri : Richardson number

    based on equation (12) from Norman et. al., 2000 (DTD paper)'''

    # See eq (2) from Louis 1979
    Ri = -(gravity * (z_u - d_0) / T_A1) * (((T_R1 - T_R0) - (T_A1 - T_A0)) / u**2)
    return Ri

def CalcU_C (u, h_C, d_0, z_0M, z_u,L):
    '''Estimates the wind speed at the canopy interface
    
    Parameters
    ----------
    u : Wubd speed measured at heigth z_u (m s-1)
    h_C : canopy height (m)
    d_0 : zero-plane displacement height
    z_0M : aerodynamic roughness lenght for momentum transport (m)
    z_u: Height of measurement of wind speeed
    L: Monin Obukhov Lenght
    
    Returns
    -------
    u_C : wind speed at the canop interface (m s-1)
    
    based on equations from appendix B of Normal et al., 1995:
    Source approach for estimating soil and vegetation energy fluxes in 
    observations of directional radiometric surface temperature'''

    from math import log
    Psi_M= CalcPsi_M((h_C - d_0)/L)
    # calcualte u_C, wind speed at the top of (or above) the canopy
    u_C = u*log ((h_C  - d_0) / z_0M)/(log ((z_u  - d_0) / z_0M)- Psi_M)
    return u_C

def CalcU_Goudriaan (u_C, h_C, LAI, leaf_width, z):
    '''Estimates the wind speed at a given height below the canopy
    
    Parameters
    ----------
    U_C : Windspeed at the canopy interface (m s-1)
    h_C : canopy height (m)
    LAI : Efective Leaf (Plant) Area Index
    leaf_width : effective leaf width size (m)
    z : heigh at which the windsped will be estimated
    
    Returns
    -------
    u_z : wind speed at height z (m s-1)
    
    based on equations from appendix B of Normal et al., 1995:
    Source approach for estimating soil and vegetation energy fluxes in 
    observations of directional radiometric surface temperature'''

    from math import exp
    a=CalcA_Goudriaan (h_C,LAI,leaf_width) # extinction factor for wind speed
    u_z = u_C * exp(-a * (1.0 - (z/h_C))) # Eq. 4.48 in Goudriaan 1977
    return u_z

def CalcA_Goudriaan (h_C,LAI,leaf_width):
    ''' Estimates the extinction coefficient factor for wind speed
    
    Parameters
    ----------    
    h_C : canopy height (m)
    LAI : Efective Leaf (Plant) Area Index
    leaf_width : effective leaf width size (m)

    Returns
    -------
    a : exctinction coefficient for wind speed through the canopy
    
    based on equations  4.49 and 4.45 of Goudriaan 1977:
    Crop micrometeorology: a simulation study'''
    # Asuming a uniform leaf area density
    from math import pi

    # Equation in Norman et al. 1995
    k3_prime=0.28
    a = k3_prime * LAI**(2./3.) * h_C**(1./3.) * leaf_width**(-1./3.)

    return a
    
def CalcU_star (u, z_u, L, d_0,z_0M, useRi=False, z_star=False):
    '''Calculates the friction velocity
    
    Parameters
    ----------
    u : wind speed above the surface (m s-1)
    z_u : wind speed measurement height (m)
    L : Obukhov stability length (m) or Richardson number, see useRi variable
    d_0 : zero-plane displacement height (m)
    z_0M : aerodynamic roughness lenght for momentum transport (m)
    useRi : Boolean variable to use the Richardson number istead of MO lenght 
        for adiabatic correction factors
    z_star : height of the roughness sublayer (RSL), optional, if used and zstar>0 
        the adiabatic correction in the RSL will be computed
 
    rearanged equation (2.54) from Brutsaert (2005): Hydrology - An Introduction (pp 47)''' 

    from math import log
    Psi_M=0.0
    Psi_M0=0.0
    Psi_M_star=0.0
    if useRi: 
        #use the approximation Ri ~ (z-d_0)./L from end of section 2.	2 from Norman et. al., 2000 (DTD paper) 
        Psi_M = CalcPsi_M(L);
        Psi_M0 = CalcPsi_M(L/(z_u - d_0)*z_0M)
    else:
        #calculate correction factors in other conditions
        if L == 0.0: L=1e-36
        Psi_M= CalcPsi_M((z_u - d_0)/L)
        Psi_M0 = CalcPsi_M(z_0M/L)
    if z_star>0 and z_u <= z_star:
        Psi_M_star=CalcPsi_M_star(z_u, L, d_0,z_0M,z_star)
    u_star = u * k / ( log( (z_u - d_0) / z_0M ) - Psi_M + Psi_M0 +Psi_M_star)
    return u_star
