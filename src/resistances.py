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

Modified on Jan 27 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This module includes functions for calculating the resistances for
heat and momentum trasnport for both One- and Two-Source Energy Balance models.
Additional functions needed in are imported from the following packages

* :doc:`meteoUtils` for the estimation of meteorological variables.
* :doc:`MOsimilarity` for the estimation of the Monin-Obukhov length and stability functions.

PACKAGE CONTENTS
================
Resistances
-----------
* :func:`CalcR_A` Aerodynamic resistance.
* :func:`CalcR_S_Kustas` [Kustas1999]_ soil resistance.
* :func:`CalcR_X_Norman` [Norman1995]_ canopy boundary layer resistance.

Stomatal conductance
--------------------
* :func:`CalcStomatalConductanceTSEB` TSEB stomatal conductance.
* :func:`CalcStomatalConductanceOSEB` OSEB stomatal conductance.
* :func:`CalcCoef_m2mmol` Conversion factor from stomatal conductance from m s-1 to mmol m-2 s-1.

Estimation of roughness
-----------------------
* :func:`CalcD_0` Zero-plane displacement height.
* :func:`CalcRoughness` Roughness for different land cover types.
* :func:`CalcZ_0M` Aerodynamic roughness lenght.
* :func:`Raupach` Roughness and displacement height factors for discontinuous canopies.

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
    ''' Zero-plane displacement height

    Calculates the zero-plane displacement height based on a 
    fixed ratio of canopy height.
    
    Parameters
    ----------
    h_C : float
        canopy height (m).
    
    Returns
    -------
    d_0 : float
        zero-plane displacement height (m).'''
    
    d_0 = h_C * 0.65
    return d_0

def CalcRoughness (LAI, hc,wc=1,landcover=11):
    ''' Surface roughness and zero displacement height for different vegetated surfaces.

    Calculates the roughness using different approaches depending we are dealing with
    crops or grasses (fixed ratio of canopy height) or shrubs and forests,depending of LAI
    and canopy shape, after [Schaudt2000]_
    
    Parameters
    ----------
    LAI : float
        Leaf (Plant) Area Index.
    hc : float
        Canopy height (m)
    wc : float, optional
        Canopy height to width ratio.
    landcover : int, optional
        landcover type, use 11 for crops, 2 for grass, 5 for shrubs,
        4 for conifer forests and 3 for broadleaved forests.
    
    Returns
    -------
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    d : float
        Zero-plane displacement height (m).
    
    References
    ----------
    .. [Schaudt2000] K.J Schaudt, R.E Dickinson, An approach to deriving roughness length
        and zero-plane displacement height from satellite data, prototyped with BOREAS data,
        Agricultural and Forest Meteorology, Volume 104, Issue 2, 8 August 2000, Pages 143-155,
        http://dx.doi.org/10.1016/S0168-1923(00)00153-2.
    '''

    from math import exp,pi
    import numpy as np
    #Needleleaf canopies
    if landcover == CONIFER:
        fc=1.-np.exp(-0.5*LAI)
        lambda_=(2./pi)*fc*wc
        #Calculation of the Raupach (1994) formulae
        z0M_factor,d_factor=Raupach(lambda_)
    #Broadleaved canopies
    elif landcover == BROADLEAVED:
        fc=1.-np.exp(-LAI)
        lambda_=fc*wc
        z0M_factor,d_factor=Raupach(lambda_)
    #Shrublands
    elif landcover == SHRUB:
        fc=1.-np.exp(-0.5*LAI)
        lambda_=fc*wc
        z0M_factor,d_factor=Raupach(lambda_)
    else:
        z0M_factor=0.125
        d_factor=0.65

    #Calculation of correction factors from  Lindroth
    fz= 0.3299*LAI**1.5+2.1713
    fd=1.-0.3991*np.exp(-0.1779*LAI)    
    # LAI <= 0
    fz[LAI <= 0] = 1.0
    fd[LAI <= 0] = 1.0
    # LAI >= 0.8775:
    i = LAI >= 0.8775
    fz[i]=1.6771*np.exp(-0.1717*LAI[i])+1.
    fd[i]=1.-0.3991*np.exp(-0.1779*LAI[i])
    
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
    MO similarity theory.
    
    Parameters
    ----------
    z_T : float
        air temperature measurement height (m).
    ustar : float
        friction velocity (m s-1).
    L : float
        Monin Obukhov Length or Richardson number for stability (see useRi variable).
    d_0 : float
        zero-plane displacement height (m).
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    z_0H : float
        aerodynamic roughness length for heat trasport (m).
    useRi : bool, optional
        boolean variable to use Richardsond number instead of the MO length.
    z_star : float or None, optional
        height of the roughness sublayer (RSL), optional, if used and zstar>0 
        the adiabatic correction in the RSL will be computed.
    
    Returns
    -------
    R_A : float
        aerodyamic resistance to heat transport in the surface layer (s m-1).

    References
    ----------        
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    import numpy as np    
    
    R_A_log = np.log( (z_T - d_0) / z_0H)
    if (useRi):
        # use the approximation Ri ~ (z-d_0)./L from end of section 2.2 from Norman et. al., 2000 (DTD paper) 				
        Psi_H = MO.CalcPsi_H(L)
        Psi_H0 = MO.CalcPsi_H(L/(z_T - d_0)*z_0H)
    else:
        #if L -> infinity, z./L-> 0 and there is neutral atmospheric stability 
        Psi_H = 0.0
        Psi_H0 = 0.0
        #other atmospheric conditions
        L[L==0] = 1e-36
        Psi_H = MO.CalcPsi_H((z_T - d_0)/L)
        Psi_H0 = MO.CalcPsi_H(z_0H/L)
    
    try:
        Psi_H_star = np.zeros(ustar.shape)
    except:
        print ustar
    #i = np.logical_and(z_star>0, z_T<=z_star)    
    #Psi_H_star[i] = MO.CalcPsi_H_star(z_T[i], L[i], d_0[i], z_0H[i], z_star[i])
    
    R_A = np.ones(ustar.shape)*float('inf')
    i = [ustar!=0]
    R_A[i] =  (R_A_log[i] - Psi_H[i] + Psi_H0[i] + Psi_H_star[i]) /(ustar[i] * k)
    return R_A
   
def CalcR_S_Kustas (u_S, deltaT):
    ''' Aerodynamic resistance at the  soil boundary layer.

    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    original equations in TSEB [Kustas1999]_.

    Parameters
    ----------
    u_S : float
        wind speed at the soil boundary layer (m s-1).
    deltaT : float
        Surface to air temperature gradient (K).
    
    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1).
   
    References
    ----------
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''
    import numpy as np    
    
    deltaT = np.maximum(deltaT, 0.0)
    R_S = np.ones(u_S.shape)*float('inf')    
    R_S[u_S>0] = 1.0/ (KN_c* deltaT[u_S>0]**(1.0/3.0)+ KN_b * u_S[u_S>0])
    return R_S

def CalcR_X_Norman(LAI, leaf_width, u_d_zm):
    ''' Estimates aerodynamic resistance at the canopy boundary layer.

    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    original equations in TSEB [Norman1995]_.

    Parameters
    ----------
    F : float
        local Leaf Area Index.
    leaf_width : float
        efective leaf width size (m).
    u_d_zm : float
        wind speed at the height of momomentum source-sink. .
    
    Returns
    -------
    R_x : float
        Aerodynamic resistance at the canopy boundary layer (s m-1).

    References
    ----------    
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''
    import numpy as np

    #C_dash = 130.0 # Original value proposed by McNaughton & Van der Hurk 1995
    C_dash_F = KN_C_dash/LAI
    R_x = np.ones(u_d_zm.shape)*float('inf')
    R_x[u_d_zm>0] = C_dash_F[u_d_zm>0]*(leaf_width/u_d_zm[u_d_zm>0])**0.5
    return R_x
  
def CalcZ_0H (z_0M,kB=2):
    '''Estimate the aerodynamic routhness length for heat trasport.
    
    Parameters
    ----------
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    kB : float
        kB parameter, default = 0.
    
    Returns
    -------
    z_0H : float
        aerodynamic roughness length for momentum transport (m).
    
    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    from math import exp
    z_OH = z_0M/exp(kB)
    return z_OH
    
def CalcZ_0M (h_C):
    ''' Aerodynamic roughness lenght.

    Estimates the aerodynamic roughness length for momentum trasport 
    as a ratio of canopy height.
    
    Parameters
    ----------
    h_C : float
        Canopy height (m).
    
    Returns
    -------
    z_0M : float
        aerodynamic roughness length for momentum transport (m).'''

    z_OM = h_C * 0.125
    return z_OM

def Raupach(lambda_):
    '''Roughness and displacement height factors for discontinuous canopies

    Estimated based on the frontal canopy leaf area, based on Raupack 1994 model,
    after [Schaudt2000]_
    
    Parameters
    ----------
    lambda_ : float
        roughness desnsity or frontal area index.
    
    Returns
    -------
    z0M_factor : float
        height ratio of roughness length for momentum transport
    d_factor : float
        height ratio of zero-plane displacement height

    References
    ----------
    .. [Schaudt2000] K.J Schaudt, R.E Dickinson, An approach to deriving roughness length
        and zero-plane displacement height from satellite data, prototyped with BOREAS data,
        Agricultural and Forest Meteorology, Volume 104, Issue 2, 8 August 2000, Pages 143-155,
        http://dx.doi.org/10.1016/S0168-1923(00)00153-2.

    '''   

    from math import exp,sqrt
    import numpy as np
    
    lambda_ = np.array(lambda_)
    z0M_factor=np.zeros(np.shape(lambda_))
    d_factor= 0.65*np.ones(np.shape(lambda_))
    
    # Calculation of the Raupach (1994) formulae
    # if lambda_ > 0.152:
    i = lambda_ > 0.152    
    z0M_factor[i] = (0.0537/(lambda_[i]**0.510))*(1.-np.exp(-10.9*lambda_[i]**0.874))+0.00368
    # else:
    z0M_factor[~i]=5.86*np.exp(-10.9*lambda_[~i]**1.12)*lambda_[~i]**1.33+0.000860
    
    # if lambda_ > 0: 
    i = lambda_ > 0    
    d_factor[i] = 1.-(1.-np.exp(-np.sqrt(15.0*lambda_[i])))/np.sqrt(15.0*lambda_[i])
    
    return z0M_factor,d_factor
