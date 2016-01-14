# This file is part of pyTSEB for calculating the canopy clumping index
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

Routines for calculating the clumping index for both randomly placed canopies and
structured row crops such as vineyards
"""
def CalcOmega0_Kustas(LAI, f_C,x_LAD=1,isLAIeff=True):
    ''' 
    Calculates the nadir viewing clmping factor
         
    Parameters
    ----------  
    LAI : Leaf Area Index, it can be either the effective LAI or the real LAI 
        , default input LAI is effective
    f_C : Apparent fractional cover, estimated from large gaps, means that
        are still gaps within the canopy to be quantified
    x_LAD : x parameter for the ellipsoildal Leaf Angle Distribution function of 
    Campbell 1988 [default=1, spherical LIDF]
    isLAIeff :  boolean varible to define whether the input LAI is effective or 
    real
    
    Returns
    ----------
    omega0 : clumping index at nadir
    
    Based on Kustas and Norman 1999. Evaluation of soil and vegetation heat flux
    predictions using a simple two-source model with radiometric temperatures
    for partial canopy cover.  Agricultural and Forest Meteorology 94 '''
    
    from math import log,exp, sqrt,radians, tan
    
    theta=0.0
    theta=radians(theta)    
    # Estimate the beam extinction coefficient based on a ellipsoidal LAD function
    # Eq. 15.4 of Campbell and Norman (1998)
    K_be=sqrt(x_LAD**2+tan(theta)**2)/(x_LAD+1.774*(x_LAD+1.182)**-0.733)
    if isLAIeff:
        F=LAI/f_C
    else: # The input LAI is actually the real LAI
        F=float(LAI)
    # Calculate the gap fraction of our canopy
    trans = f_C*exp(-K_be * F)+(1.0-f_C)
    if trans<=0:
        trans=1e-36
    # and then the nadir clumping factor
    omega0 = -log(trans)/(LAI*K_be)
    return omega0

def CalcOmega_Kustas(omega0,theta,wc=1):
    '''
    Estimates the clumpnig index for a given incidence angle assuming randomnly placed canopies
    
    Parameters
    ----------
    theta: incidence angle (degrees)
    D :  canopy witdth to height ratio, [default = 1]

    Based on Kustas and Norman 1999. Evaluation of soil and vegetation heat flux
    predictions using a simple two-source model with radiometric temperatures
    for partial canopy cover.  Agricultural and Forest Meteorology 94,
    after Campbell and Norman 1998. An Introduction to Environmental Biophysics.
    Springer, New York, 286 pp
    '''
    
    from math import exp,radians
    wc=1.0/wc
    omega = omega0 / (omega0 + (1.0 - omega0) * exp(-2.2 * (radians(theta))**(3.8 - 0.46 * wc)))
    return omega
