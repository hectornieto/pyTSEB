# This file is part of pyTSEB for estimating the resistances to momentum and heat transport 
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

Routines for calculating the resistances for heat and momentum trasnport
"""

#==============================================================================
# List of constants used in TSEB model and sub-routines   
#==============================================================================
# Land Cover Classes
CROP=11
GRASS=2
SHRUB=5
CONIFER=4
BROADLEAVED=3
# von Karman's constant
k=0.4 
#acceleration of gravity (m s-2)
gravity=9.8
# Universal gas constant (kPa m3 mol-1 K-1)
R_u=0.0083144

KN_b = 0.012 # Value propoesd in Kustas et al 1999
KN_c = 0.0025 # Coefficient from Norman et al. 1995
KN_C_dash = 90.0 # value proposed in Norman et al. 1995

import MOsimilarity as MO
import meteoUtils as met

def CalcD_0 (h_C):
    ''' Calculates the zero-plane displacement height based on a 
    fixed ratio of canopy height
    
    Parameters
    ----------
    h_C : canopy height (m)
    
    Returns
    -------
    d_0 : zero-plane displacement height (m)'''
    
    d_0 = h_C * 0.65
    return d_0

def CalcRoughness (LAI, hc,wc=1,landcover=11):
    ''' Retrieves Surface roughness and zero displacement height for vegetated surfaces
    
    Parameters
    ----------
    LAI : Leaf (Plant) Area Index
    hc : Canopy height (m)
    wc : Canopy height to width ratio
    landcover : landcover type, use 11 for crops, 2 for grass, 5 for shrubs,
        4 for conifer forests and 3 for broadleaved forests.
    
    Returns
    -------
    z_0M : aerodynamic roughness length for momentum trasport (m)
    d : Zero-plane displacement height (m)
    
    based on Schaudt and Dickinson (200)
    Agricultural and Forest Meteorology 104 (2000) 143-155'''
    from math import exp,pi
    #Needleleaf canopies
    if landcover == CONIFER:
        fc=1.-exp(-0.5*LAI)
        lambda_=(2./pi)*fc*wc
        #Calculation of the Raupach (1994) formulae
        z0M_factor,d_factor=Raupach(lambda_)
    #Broadleaved canopies
    elif landcover == BROADLEAVED:
        fc=1.-exp(-LAI)
        lambda_=fc*wc
        z0M_factor,d_factor=Raupach(lambda_)
    #Shrublands
    elif landcover == SHRUB:
        fc=1.-exp(-0.5*LAI)
        lambda_=fc*wc
        z0M_factor,d_factor=Raupach(lambda_)
    else:
        z0M_factor=0.125
        d_factor=0.65

    #Calculation of correction factors from  Lindroth
    if LAI <= 0:
        fz= 1.0
        fd= 1.0
    elif LAI < 0.8775:
        fz= 0.3299*LAI**1.5+2.1713
        fd=1.-0.3991*exp(-0.1779*LAI)
    else:
        fz=1.6771*exp(-0.1717*LAI)+1.
        fd=1.-0.3991*exp(-0.1779*LAI)
    #Application of the correction factors to roughness and displacement height
    z0M_factor=z0M_factor*fz
    d_factor=d_factor*fd
    if landcover == CROP or landcover == GRASS:
        z0M_factor=1./8.
        d_factor=0.65
    #Calculation of rouhgness length
    z_0M=z0M_factor*hc
    #Calculation of zero plane displacement height
    d=d_factor*hc
    return z_0M, d

def CalcR_A (z_T, ustar, L, d_0, z_0H, useRi=False, z_star=False):
    ''' Estimates the aerodynamic resistance to heat transport based on the
    MO similarity theory
    
    Parameters
    ----------
    z_T : air temperature measurement height (m)
    ustar : friction velocity (m s-1)
    L : Monin Obukhov Length or Richardson number for stability (see useRi variable)
    d_0 : zero-plane displacement height (m)
    z_0M : aerodynamic roughness length for momentum trasport (m)
    z_0H : aerodynamic roughness length for heat trasport (m)
    useRi : boolean variable to use Richardsond number instead of the MO length
    z_star : height of the roughness sublayer (RSL), optional, if used and zstar>0 
        the adiabatic correction in the RSL will be computed
    
    Returns
    -------
    R_A : aerodyamic resistance to heat transport in the surface layer (s m-1)

    based on equation (10) from Norman et. al., 2000 (DTD paper)'''

    from math import log
    if ustar==0:return float('inf')
    Psi_H_star=0.0
    R_A_log = log( (z_T - d_0) / z_0H)
    if (useRi):
        # use the approximation Ri ~ (z-d_0)./L from end of section 2.2 from Norman et. al., 2000 (DTD paper) 				
        Psi_H = MO.CalcPsi_H(L)
        Psi_H0 = MO.CalcPsi_H(L/(z_T - d_0)*z_0H)
    else:
        #if L -> infinity, z./L-> 0 and there is neutral atmospheric stability 
        Psi_H = 0.0
        Psi_H0 = 0.0
        #other atmospheric conditions
        if L == 0.0:L=1e-36
        Psi_H = MO.CalcPsi_H((z_T - d_0)/L)
        Psi_H0 = MO.CalcPsi_H(z_0H/L)
    if z_star>0 and z_T<=z_star:
        Psi_H_star=MO.CalcPsi_H_star(z_T, L, d_0,z_0H,z_star)
    R_A =  (R_A_log - Psi_H + Psi_H0+Psi_H_star) /(ustar * k)
    return R_A
   
def CalcR_S_Kustas (u_S, deltaT):
    ''' Estimates aerodynamic resistance at the  soil boundary layer

    Parameters
    ----------
    u_S : wind speed at the soil boundary layer (s m-1)
    deltaT : Surface to air temperature gradient (K)
    
    Returns
    -------
    R_S : Aerodynamic resistance at the  soil boundary layer (s m-1)
    
    based on Kustas and Norman 1999'''
    if u_S==0:return float('inf')

    if deltaT<0.0:
        deltaT = 0.0
    R_S = 1.0/ (KN_c* deltaT**(1.0/3.0)+ KN_b * u_S)
    return R_S

def CalcR_X_Norman(LAI, leaf_width, u_d_zm):
    ''' Estimates aerodynamic resistance at the canopy boundary layer

    Parameters
    ----------
    LAI : Leaf (Plant) Area Index
    leaf_width : efective leaf width size (m)
    u_d_zm : wind speed at d+zm
    
    Returns
    -------
    R_x : Aerodynamic resistance at the canopy boundary layer (s m-1)
    
    based on eq. A.8 in Norman et al 1995'''
    if u_d_zm==0:return float('inf')
    #C_dash = 130.0 # Original value proposed by McNaughton & Van der Hurk 1995
    C_dash_F = KN_C_dash/LAI
    R_x = C_dash_F*(leaf_width/u_d_zm)**0.5
    return R_x
  
def CalcZ_0H (z_0M,kB=2):
    '''Estimate the aerodynamic routhness length for heat trasport
    
    Parameter
    ---------
    z_0M : aerodynamic roughness lenght for momentum transport (m)
    kB : kB parameter, default = 2
    
    Results
    -------
    z_0H : aerodynamic roughness lenght for momentum transport (m)

    based on equation from section 2.2 of Norman et. al., 2000 (DTD paper)'''
    from math import exp
    z_OH = z_0M/exp(kB)
    return z_OH
    
def CalcZ_0M (h_C):
    '''Estimate the aerodynamic routhness length for momentum trasport 
    as a ratio of canopy height
    
    Parameter
    ---------
    h_C : Canopy height (m)
    
    Results
    -------
    z_0M : aerodynamic roughness lenght for momentum transport (m)'''
    z_OM = h_C * 0.125
    return z_OM

def Raupach(lambda_):
    '''Estimate the roughness and displacement height factors based on Raupack 1994 
    
    Parameter
    ---------
    lambda_ : roughness desnsity or frontal area index
    
    Results
    -------
    z_0M : aerodynamic roughness lenght for momentum transport (m)'''   
    from math import exp,sqrt
    z0M_factor=0.125
    d_factor=0.65
    # Calculation of the Raupach (1994) formulae
    if lambda_ > 0.152:
        z0M_factor=(0.0537/(lambda_**0.510))*(1.-exp(-10.9*lambda_**0.874))+0.00368
    else:
        z0M_factor=5.86*exp(-10.9*lambda_**1.12)*lambda_**1.33+0.000860
    if lambda_ > 0: d_factor=1.-(1.-exp(-sqrt(15.0*lambda_)))/sqrt(15.0*lambda_)
    return z0M_factor,d_factor
