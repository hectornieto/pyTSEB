# This file is part of pyTSEB for calculating the net radiation and its divergence
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
@author: Hector Nieto (hnieto@ias.csic.es).

Modified on Jan 27 2016
@author: Hector Nieto (hnieto@ias.csic.es).

DESCRIPTION
===========
This package contains functions for estimating the net shortwave and longwave radiation
for soil and canopy layers. Additional packages needed are.

* :doc:`meteoUtils` for the estimation of meteorological variables.

PACKAGE CONTENTS
================
* :func:`CalcDifuseRatio` estimation of fraction of difuse shortwave radiation.
* :func:`CalcEmiss_atm` Atmospheric emissivity.
* :func:`CalcKbe_Campbell` Beam extinction coefficient.
* :func:`CalcLnKustas` Net longwave radiation for soil and canopy layers.
* :func:`CalcRnOSEB` Net radiation in a One Source Energy Balance model.
* :func:`CalcSnCampbell` Net shortwave radiation. 
'''


#==============================================================================
# List of constants used in the netRadiation Module
#==============================================================================
#Stephan Boltzmann constant (W m-2 K-4)
sb=5.670373e-8 

import meteoUtils as met

def CalcDifuseRatio(Sdn,sza,Wv=1,press=1013.25):
    '''Fraction of difuse shortwave radiation.

    Partitions the incoming solar radiation into PAR and non-PR and
    diffuse and direct beam component of the solar spectrum.
    
    Parameters
    ----------
    Sdn : float
        Incoming shortwave radiation (W m-2).
    sza : float
        Solar Zenith Angle (degrees).
    Wv : float, optional
        Total column precipitable water vapour (g cm-2), default 1 g cm-2.
    press : float, optional
        atmospheric pressure (mb), default at sea level (1013mb).
        
    Returns
    -------
    difvis : float
        diffuse fraction in the visible region.
    difnir : float
        diffuse fraction in the NIR region.
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.
    
    References
    ----------
    .. [Weiss1985] Weiss and Norman (1985) Partitioning solar radiation into direct and diffuse,
        visible and near-infrared components, Agricultural and Forest Meteorology,
        Volume 34, Issue 2, Pages 205-213,
        http://dx.doi.org/10.1016/0168-1923(85)90020-6.
    '''
    from math import radians, cos,exp, log10
    coszen=abs(cos(radians(sza)))
    #Calculate potential (clear-sky) visible and NIR solar components
    # Weiss & Norman 1985
    #Correct for curvature of atmos in airmas (Kasten and Young,1989)
    if sza >90:
        difvis,difnir, fvis,fnir=[1.0,1.0, 0.4,0.6]
        return difvis,difnir, fvis,fnir
    else:
        #airmas=1.0/(abs(coszen)+0.50572*radians(96.07995-sza)**-1.6364)
        airmas=1.0/coszen
    #Visible PAR/NIR direct beam radiation
    Rdirvis=600.*exp(-.185*airmas)*coszen                                   #Eq. 1
    Rdirvis=max(0,Rdirvis)
    w=1320.0*Wv*10**(-1.195+.4459*log10(airmas)-.0345*log10(airmas)**2)     #Eq. 6
    Rdirnir=(720.*exp(-0.06*(press/1313.25)*airmas)-w)*coszen                               #Eq. 4
    Rdirnir=max(0,Rdirnir)    
    # Potential diffuse radiation
    Rdifvis=0.4*(600.0-Rdirvis)*coszen                                      #Eq. 3
    Rdifnir=0.6*(720.0-Rdirvis-w)*coszen                                    #Eq. 5
    Rdifvis=max(0,Rdifvis)
    Rdifnir=max(0,Rdifnir)      
    #Potential total solar radiation
    potvis=Rdirvis+Rdifvis
    if potvis<=0:potvis=1e-6
    potnir=Rdirnir+Rdifnir
    if potnir<=0:potnir=1e-6
    fclear=Sdn/(potvis+potnir)
    fclear=min(1.0,fclear)
    #Partition SDN into VIS and NIR      
    fvis=potvis/(potvis+potnir)                                             #Eq. 7
    fnir=potnir/(potvis+potnir)                                            #Eq. 8
    fvis=max(0,fvis)
    fvis=min(1,fvis)
    fnir=1.0-fvis
    #Estimate direct beam and diffuse fractions in VIS and NIR wavebands
    ratiox=fclear
    if fclear > 0.9:ratiox=.9
    dirvis=(Rdirvis/potvis)*(1.-((.9-ratiox)/.7)**.6667)                    #Eq. 11
    if fclear > 0.88:ratiox=.88
    dirnir=(Rdirnir/potnir)*(1.-((.88-ratiox)/.68)**.6667)                  #Eq. 12
    dirvis=max(0.0,dirvis)
    dirnir=max(0.0,dirnir)
    dirvis=min(1.0,Rdirvis/potvis,dirvis)
    dirnir=min(1.0,Rdirnir/potnir,dirnir)
    if dirvis < 0.01 and dirnir > 0.01:dirvis=.011
    if dirnir < 0.01 and dirvis > 0.01:dirnir=.011
    difvis=1.0-dirvis
    difnir=1.0-dirnir
    return difvis,difnir, fvis,fnir

def CalcEmiss_atm(ea,Ta_K):
    '''Atmospheric emissivity
    
    Estimates the effective atmospheric emissivity for clear sky.

    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    Ta_K : float
        air temperature (Kelvin).
    
    Returns
    -------
    emiss_air : float
        effective atmospheric emissivity.

    References
    ----------    
    .. [Brutsaert1975] Brutsaert, W. (1975) On a derivable formula for long-wave radiation
        from clear skies, Water Resour. Res., 11(5), 742-744,
        htpp://dx.doi.org/10.1029/WR011i005p00742.'''

    emiss_air=1.24*(ea/Ta_K)**(1./7.)
    return emiss_air

def CalcKbe_Campbell(theta,x_LAD=1):
    ''' Beam extinction coefficient

    Calculates the beam extinction coefficient based on [Campbell1998]_ ellipsoidal
    leaf inclination distribution function.
    
    Parameters
    ----------
    theta : float
        incidence zenith angle (degrees).
    x_LAD : float, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function, 
        use x_LAD=1 for a spherical LAD.
    
    Returns
    -------
    K_be : float
        beam extinction coefficient.
    x_LAD: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function, 
        use x_LAD=1 for a spherical LAD.
    
    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    '''

    import numpy as np  
    theta=np.radians(theta)
    K_be=np.sqrt(x_LAD**2+np.tan(theta)**2)/(x_LAD+1.774*(x_LAD+1.182)**-0.733)
    return K_be
    
def CalcLnKustas (T_C, T_S, Lsky, LAI, emisVeg, emisGrd,x_LAD=1):
    ''' Net longwave radiation for soil and canopy layers

    Estimates the net longwave radiation for soil and canopy layers unisg based on equation 2a
    from [Kustas1999]_ and incorporated the effect of the Leaf Angle Distribution based on [Campbell1998]_
    
    Parameters
    ----------
    T_C : float
        Canopy temperature (K).
    T_S : float
        Soil temperature (K).
    Lsky : float
        Downwelling atmospheric longwave radiation (w m-2).
    LAI : float
        Effective Leaf (Plant) Area Index.
    emisVeg : float
        Broadband emissivity of vegetation cover.
    emisGrd : float
        Broadband emissivity of soil.
    x_LAD: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function, 
        use x_LAD=1 for a spherical LAD.

    Returns
    -------
    L_nC : float
        Net longwave radiation of canopy (W m-2).
    L_nS : float
        Net longwave radiation of soil (W m-2).   

    References
    ----------
    .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    from math import sqrt,cos,sin,radians
    import numpy as np    
    
    # Integrate to get the diffuse transmitance
    taud=0
    for angle in range(0,90,5):
        akd=CalcKbe_Campbell(angle,x_LAD)# Eq. 15.4
        taub=np.exp(-akd*LAI)
        taud = taud+taub*cos(radians(angle))*sin(radians(angle))*radians(5)
    taud=2.0*taud
    #D I F F U S E   C O M P O N E N T S
    #Diffuse light canopy reflection coefficients  for a deep canopy	
    akd=-np.log(taud)/LAI
    ameanl=emisVeg    
    taudl=np.exp(-sqrt(ameanl)*akd*LAI)    #Eq 15.6
    # calculate long wave emissions from canopy, soil and sky
    L_C = emisVeg*met.CalcStephanBoltzmann(T_C)
    L_S = emisGrd*met.CalcStephanBoltzmann(T_S)
    # calculate net longwave radiation divergence of the soil
    L_nS = taudl*Lsky + (1.0-taudl)*L_C - L_S
    L_nC = (1.0-taudl)*(Lsky + L_S - 2.0*L_C)
    return L_nC,L_nS

def CalcRnOSEB(Sdn,Lsky, T_R, emis, albedo):
    ''' Net radiation in a One Source Energy Balance model

    Estimates surface net radiation assuming a single `big leaf` layer.
        
    Parameters
    ----------
    Sdn : float
        incoming shortwave radiation (W m-2).
    Lsky : float
        Incoming longwave radiation (W m-2).   
    T_R : float
        Radiometric surface temperature (K).
    emis : float
        Broadband emissivity.
    albedoVeg : float
        Broadband short wave albedo.
    
    Returns
    -------
    R_s : float
        Net shortwave radiation (W m-2).
    R_l : float
        Net longwave radiation (W m-2).
    '''
    # outgoing shortwave radiation
    R_sout = Sdn * albedo
    # outgoing long wave radiation
    R_lout = emis * sb * (T_R)**4 + Lsky * (1.0 - emis)
    R_s = Sdn-R_sout
    R_l = Lsky-R_lout
    return R_s, R_l
    
def CalcSnCampbell (LAI, sza, Sdn_dir, Sdn_dif, fvis,fnir, rho_leaf_vis,
                    tau_leaf_vis,rho_leaf_nir, tau_leaf_nir, rsoilv, rsoiln,x_LAD=1, LAI_eff=None):
    ''' Net shortwave radiation 

    Estimate net shorwave radiation for soil and canopy below a canopy using the [Campbell1998]_
    Radiative Transfer Model, and implemented in [Kustas1999]_
    
    Parameters
    ----------
    LAI : float
        Effective Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    Sdn_dir : float
        Broadband incoming beam shortwave radiation (W m-2).
    Sdn_dir : float
        Broadband incoming beam shortwave radiation (W m-2).
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.
    rho_leaf_vis : float
        Broadband leaf bihemispherical reflectance in the visible region (400-700nm).
    tau_leaf_vis : float
        Broadband leaf bihemispherical transmittance in the visible region (400-700nm).
    rho_leaf_nir : float
        Broadband leaf bihemispherical reflectance in the NIR region (700-2500nm).
    tau_leaf_nir : float
        Broadband leaf bihemispherical transmittance in the NIR region (700-2500nm).
    rsoilv : float
        Broadband soil bihemispherical reflectance in the visible region (400-700nm).
    rsoiln : float
        Broadband soil bihemispherical reflectance in the NIR region (700-2500nm).
    x_LAD : float,  optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of 
        Campbell 1988 [default=1, spherical LIDF].
    LAI_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies
    Returns
    -------
    Rn_sw_veg : float
        Canopy net shortwave radiation (W m-2).
    Rn_sw_soil : float
        Soil net shortwave radiation (W m-2).

    References
    ----------    
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''
    
    from math import radians, cos, sin, sqrt
    import numpy as np
    #calculate aborprtivity
    ameanv = 1.0-rho_leaf_vis-tau_leaf_vis
    ameann = 1.0-rho_leaf_nir-tau_leaf_nir
    # Calculate canopy beam extinction coefficient
    #Modification to include other LADs
    if type(LAI_eff)==type(None):
        LAI_eff=np.asarray(LAI)
    else:
        LAI_eff=np.asarray(LAI_eff)
    # Integrate to get the diffuse transmitance
    taud=0
    for angle in range(0,90,5):
        akd=CalcKbe_Campbell(angle,x_LAD) # Eq. 15.4
        taub=np.exp(-akd*LAI)
        taud = taud+taub*cos(radians(angle))*sin(radians(angle))*radians(5)
    taud=2.0*taud
    #D I F F U S E   C O M P O N E N T S
    #Diffuse light canopy reflection coefficients  for a deep canopy	
    #akd=-0.0683*log(LAI)+0.804                  # Fit to Fig 15.4 for x=1
    akd=-np.log(taud)/LAI
    rcpyn=(1.0-sqrt(ameann))/(1.0+sqrt(ameann)) # Eq 15.7   
    rcpyv=(1.0-sqrt(ameanv))/(1.0+sqrt(ameanv))
    rdcpyn=2.0*akd*rcpyn/(akd+1.0)              #Eq 15.8      
    rdcpyv=2.0*akd*rcpyv/(akd+1.0) 
    #Diffuse canopy transmission coeff (visible) 				
    expfac = sqrt(ameanv)*akd*LAI
    xnum = (rdcpyv*rdcpyv-1.0)*np.exp(-expfac)
    xden = (rdcpyv*rsoilv-1.0)+rdcpyv*(rdcpyv-rsoilv)*np.exp(-2.0*expfac)
    taudv = xnum/xden                           #Eq 15.11
    #Diffuse canopy transmission coeff (NIR) 				
    expfac = sqrt(ameann)*akd*LAI;
    xnum = (rdcpyn*rdcpyn-1.0)*np.exp(-expfac);
    xden = (rdcpyn*rsoiln-1.0)+rdcpyn*(rdcpyn-rsoiln)*np.exp(-2.0*expfac)
    taudn = xnum/xden                           #Eq 15.11
    #Diffuse radiation surface albedo for a generic canopy
    fact=((rdcpyn-rsoiln)/(rdcpyn*rsoiln-1.0))*np.exp(-2.0*sqrt(ameann)*akd*LAI)   #Eq 15.9
    albdn=(rdcpyn+fact)/(1.0+rdcpyn*fact)
    fact=((rdcpyv-rsoilv)/(rdcpyv*rsoilv-1.0))*np.exp(-2.0*sqrt(ameanv)*akd*LAI)   #Eq 15.9
    albdv=(rdcpyv+fact)/(1.0+rdcpyv*fact)
    #B E A M   C O M P O N E N T S
    #Direct beam extinction coeff (spher. LAD)  
    akb=CalcKbe_Campbell(sza,x_LAD) # Eq. 15.4
    #Direct beam canopy reflection coefficients for a deep canopy
    rcpyn=(1.0-sqrt(ameann))/(1.0+sqrt(ameann))                                 #Eq 15.7   
    rcpyv=(1.0-sqrt(ameanv))/(1.0+sqrt(ameanv))
    rbcpyn=2.0*akb*rcpyn/(akb+1.0)                                              #Eq 15.8      
    rbcpyv=2.0*akb*rcpyv/(akb+1.0); 
    fact=((rbcpyn-rsoiln)/(rbcpyn*rsoiln-1.0))*np.exp(-2.0*sqrt(ameann)*akb*LAI_eff)   #Eq 15.9
    albbn=(rbcpyn+fact)/(1.0+rbcpyn*fact)
    fact=((rbcpyv-rsoilv)/(rbcpyv*rsoilv-1.0))*np.exp(-2.0*sqrt(ameanv)*akb*LAI_eff)   #Eq 15.9
    albbv=(rbcpyv+fact)/(1.0+rbcpyv*fact)
    #Weighted average albedo 
    albedo_dir=fvis*albbv+fnir*albbn
    albedo_dif=fvis*albdv+fnir*albdn
    #Direct beam+scattered canopy transmission coeff (visible) 				
    expfac = sqrt(ameanv)*akb*LAI_eff
    xnum = (rbcpyv*rbcpyv-1.0)*np.exp(-expfac)
    xden = (rbcpyv*rsoilv-1.0)+rbcpyv*(rbcpyv-rsoilv)*np.exp(-2.0*expfac)
    taubtv = xnum/xden                                                          #Eq 15.11
    #Direct beam+scattered canopy transmission coeff (NIR) 				
    expfac = sqrt(ameann)*akb*LAI_eff
    xnum = (rbcpyn*rbcpyn-1.0)*np.exp(-expfac)
    xden = (rbcpyn*rsoiln-1.0)+rbcpyn*(rbcpyn-rsoiln)*np.exp(-2.0*expfac)
    taubtn = xnum/xden                                                          #Eq 15.11
    tau_dir=fvis*taubtv+fnir*taubtn
    tau_dif=fvis*taudv+fnir*taudn
    albedosoil=fvis*rsoilv+fnir*rsoiln
    Rn_sw_veg=(1.0-tau_dir)*(1.0-albedo_dir)*Sdn_dir+(1.0-tau_dif)*(1.0-albedo_dif)*Sdn_dif
    Rn_sw_soil=tau_dir*(1.0-albedosoil)*Sdn_dir+tau_dif*(1.0-albedosoil)*Sdn_dif
    return Rn_sw_veg, Rn_sw_soil
