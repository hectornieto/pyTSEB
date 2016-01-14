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

Modified on Dec 30 2015
@author: Hector Nieto (hnieto@ias.csic.es)

Routines for ancillary calculations in TSEB
'''

#==============================================================================
# List of constants used in Meteorological computations  
#==============================================================================
#Stephan Boltzmann constant (W m-2 K-4)
sb=5.670373e-8 
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd=1003.5 
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv=1865
#ratio of the molecular weight of water vapor to dry air
epsilon=0.622 
#Psicrometric Constant kPa K-1
psicr=0.0658    
# gas constant for dry air, J/(kg*degK)
R_d=287.04


def CalcC_p (p, ea):
    ''' Calculates the heat capacity of air at constant pressure
    
    Parameters
    ----------
    p : total air pressure (dry air + water vapour) (mb)
    ea : water vapor pressure at reference height above canopy (mb)
    
    Returns
    -------
    c_p : heat capacity of (moist) air at constant pressure (J kg-1 K-1)
    
    based on equation (6.1) from Maarten Ambaum (2010): 
    Thermal Physics of the Atmosphere (pp 109)'''

    # first calculate specific humidity, rearanged eq (5.22) from Maarten Ambaum (2010), (pp 100)
    q = epsilon * ea / (p + (epsilon - 1.0)*ea)
    # then the heat capacity of (moist) air
    c_p = (1.0-q)*c_pd + q*c_pv
    return c_p
    
def CalcLambda(Ta_K):
    '''Calculates the latent heat of vaporization
    
    Parameter
    ---------
    Ta_K : Air temperature (Kelvin)
    
    Results
    -------
    Lambda : Latent heat of vaporisation (MJ kg-1)
    
    based on Eq. 3-1 Allen FAO98 '''
    Lambda = 2.501 - (2.361e-3* (Ta_K-273.15) ) 
    return Lambda

def CalcPressure(z):
    ''' Calculates the barometric pressure above sea level
    
    Parameter
    ---------
    z: height above sea level (m)
    
    Returns
    -------
    p: air pressure (mb)'''
    
    p=1013.25*(1.0-2.225577e-5*z)**5.25588
    return p

def CalcPsicr(p,Lambda):
    ''' Calculates the psicrometric constant
     
    Parameter
    ---------
    p : atmopheric pressure (mb)
    Lambda : latent heat of vaporzation (MJ kg-1)
    
    Returns
    -------
    psicr : Psicrometric constant (mb C-1)'''
    
    psicr=0.00163*p/(Lambda)
    return psicr
    
def CalcRho (p, ea, Ta_K):
    '''Calculates the density of air
    
    Parameters
    ----------
    p : total air pressure (dry air + water vapour) (mb)
    ea : water vapor pressure at reference height above canopy (mb)
    Ta_K : air temperature at reference height (Kelvin)
    
    Returns
    -------
    rho : density of air (kg m-3)

    based on equation (2.6) from Brutsaert (2005): Hydrology - An Introduction (pp 25)'''
    
    # p is multiplied by 100 to convert from mb to Pascals
    rho = ((p*100.0)/(R_d * Ta_K )) * (1.0 - (1.0 - epsilon) * ea / p)
    return rho

def CalcStephanBoltzmann(T_K):
    '''Calculates the total energy radiated by a blackbody
    
    Parameter
    ---------
    T_K : body temperature (Kelvin)
    
    Returns
    -------
    M : Emitted radiance (W m-2)'''
    
    M=sb*T_K**4
    return M
    
def CalcTheta_s (xlat,xlong,stdlng,doy,year,ftime):
    '''Calculates the Sun Zenith Angle (SZA)
    
    Parameters
    ----------
    xlat - latitude of the site (degrees)
    xlong - longitude of the site (degrees)
    stdlng - central longitude of the time zone of the site (degrees)
    doy - day of year of measurement (1-366)
    year - year of measurement 
    ftime - time of measurement (decimal hours)
    
    Returns
    -------
    theta_s : Sun Zenith Angle (degrees)
    
    Adpopted from Martha Anderson's fortran code for ALEXI which in turn was based on Cupid'''
    
    from math import pi, radians,sin, cos, asin, acos, degrees
    pid180 = pi/180
    pid2 = pi/2.0
    # Latitude computations
    xlat=radians(xlat)
    sinlat=sin(xlat)
    coslat=cos(xlat)
    # Declination computations
    kday=(year-1977.0)*365.0+doy+28123.0
    xm=radians(-1.0+0.9856*kday)
    delnu=2.0*0.01674*sin(xm)+1.25*0.01674*0.01674*sin(2.0*xm)
    slong=radians((-79.8280+0.9856479*kday))+delnu
    decmax=sin(radians(23.44))
    decl=asin(decmax*sin(slong))
    sindec=sin(decl)
    cosdec=cos(decl)
    eqtm=9.4564*sin(2.0*slong)/cosdec-4.0*delnu/pid180
    eqtm=eqtm/60.0
    # Get sun zenith angle
    timsun=ftime    #MODIS time is already solar time
    hrang=(timsun-12.0)*pid2/6.0
    theta_s = acos(sinlat*sindec+coslat*cosdec*cos(hrang))
    # if the sun is below the horizon just set it slightly above horizon
    theta_s = min(theta_s, pid2-0.0000001)
    theta_s=degrees(theta_s)
    return theta_s

def Get_SunAngles(lat,lon,stdlon,doy,ftime):
    '''Calculates the Sun Zenith and Azimuth Angles (SZA & SAA)
    
    Parameters
    ----------
    lat - latitude of the site (degrees)
    long - longitude of the site (degrees)
    stdlon - central longitude of the time zone of the site (degrees)
    doy - day of year of measurement (1-366)
    ftime - time of measurement (decimal hours)
    
    Returns
    -------
    sza : Sun Zenith Angle (degrees)
    saa : Sun Azimuth Angle (degrees)
    '''
  
    from math import pi, sin, cos, asin, acos, radians, degrees
    # Calculate declination
    declination=0.409*sin((2.0*pi*doy/365.0)-1.39)
    EOT=0.258*cos(declination)-7.416*sin(declination)-3.648*cos(2.0*declination)-9.228*sin(2.0*declination)
    LC=(stdlon-lon)/15.
    time_corr=(-EOT/60.)+LC
    solar_time=ftime-time_corr
    # Get the hour angle
    w=(solar_time-12.0)*15.
    # Get solar elevation angle
    sin_thetha=cos(radians(w))*cos(declination)*cos(radians(lat))+sin(declination)*sin(radians(lat))
    sun_elev=asin(sin_thetha)
    # Get solar zenith angle
    sza=pi/2.0-sun_elev
    sza=degrees(sza)
    # Get solar azimuth angle
    cos_phi=(sin(declination)*cos(radians(lat))-cos(radians(w))*cos(declination)*sin(radians(lat)))/cos(sun_elev)
    if w <= 0.0:
        saa=degrees(acos(cos_phi))
    else:
        saa=360.-degrees(acos(cos_phi))
    return sza,saa

def CalcVaporPressure(T_K):
    '''Calculate the saturation water vapour pressure
    
    Parameter
    ---------
    T_K : temperature (K)
    
    Results
    -------
    ea : saturation water vapour pressure (mb)'''
   
    from math import exp
    T_C=T_K-273.15
    ea= 6.112 * exp((17.67*T_C)/(T_C + 243.5))
    return ea
    
def CalcDeltaVaporPressure(T_K):
    '''Calculate the slope of saturation water vapour pressure
    
    Parameter
    ---------
    T_K : temperature (K)
    
    Results
    -------
    s : slope of the saturation water vapour pressure (kPa K-1)'''
   
    from math import exp
    T_C=T_K-273.15
    s= 4098.0 * (0.6108*exp(17.27*T_C/(T_C+237.3)))/((T_C+237.3)**2)
    return s
