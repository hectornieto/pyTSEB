# This file is part of pyTSEB for running different TSEB models
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

Modified on Jan 27 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the main routines inherent of Two Source Energy Balance `TSEB` models.
Additional functions needed in TSEB, such as computing of net radiation or estimating the
resistances to heat and momentum transport are imported.

* :doc:`netRadiation` for the estimation of net radiation and radiation partitioning.
* :doc:`ClumpingIndex` for the estimatio of canopy clumping index.
* :doc:`meteoUtils` for the estimation of meteorological variables.
* :doc:`resistances` for the estimation of the resistances to heat and momentum transport.
* :doc:`MOsimilarity` for the estimation of the Monin-Obukhov length and MOST-related variables.

PACKAGE CONTENTS
================

TSEB models
-----------
* :func:`TSEB_2T` TSEB using derived/measured canopy and soil component temperatures.
* :func:`TSEB_PT` Priestley-Taylor TSEB using a single observation of composite radiometric temperature.
* :func:`DTD` Dual-Time Differenced TSEB using composite radiometric temperatures at two times: early morning and near afternoon.

OSEB models
-----------
* :func:`OSEB`. One Source Energy Balance Model.
* :func:`OSEB_BelowCanopy`. One Source Energy Balance Model of baresoil/understory beneath a canopy.
* :func:`OSEB_Canopy`. One Source Energy Balance Model of a very dense canopy, i.e. `Big-leaf` model.

Ancillary functions
-------------------
* :func:`CalcFthetaCampbell`. Gap fraction estimation.
* :func:`CalcG_TimeDiff`. Santanello & Friedl (2003) [Santanello2003]_ soil heat flux model.
* :func:`CalcG_Ratio`. Soil heat flux as a fixed fraction of net radiation [Choudhury1987]_.
* :func:`CalcH_C`. canopy sensible heat flux in a parallel resistance network.
* :func:`CalcH_C_PT`. Priestley- Taylor Canopy sensible heat flux.
* :func:`CalcH_DTD_parallel`. Priestley- Taylor Canopy sensible heat flux for DTD and resistances in parallel.
* :func:`CalcH_DTD_series`. Priestley- Taylor Canopy sensible heat flux for DTD and resistances in series.
* :func:`CalcH_S`. Soil heat flux with resistances in parallel.
* :func:`CalcT_C`. Canopy temperature form composite radiometric temperature.
* :func:`CalcT_C_Series.` Canopy temperature from canopy sensible heat flux and resistance in series.
* :func:`CalcT_CS_Norman`. Component temperatures from dual angle composite radiometric tempertures.
* :func:`CalcT_CS_4SAIL`. Component temperatures from dual angle composite radiometric tempertures. Using 4SAIl for the inversion.
* :func:`Get4SAILEmissionParam`. Effective surface reflectance, and emissivities for soil and canopy using 4SAIL.
* :func:`CalcT_S`. Soil temperature from form composite radiometric temperature.
* :func:`CalcT_S_Series`. Soil temperature from soil sensible heat flux and resistance in series.
'''

import meteoUtils as met
import resistances as res
import MOsimilarity as MO
import netRadiation as rad
import ClumpingIndex as CI
#==============================================================================
# List of constants used in TSEB model and sub-routines   
#==============================================================================
#Change threshold in  Monin-Obukhov lengh to stop the iterations
L_thres=0.00001
# Change threshold in  friction velocity to stop the iterations
u_thres=0.00001
# mimimun allowed friction velocity    
u_friction_min=0.01;
#Maximum number of interations
ITERATIONS=1000
# Priestly - Taylor cooficient
defaultAlpha=1.26
#ratio of the molecular weight of water vapor to dry air
epsilon=0.622 
#Psicrometric Constant kPa K-1
psicr=0.0658    
#Stephan Boltzmann constant (W m-2 K-4)
sb=5.670373e-8 
# kB coefficient
kB=0.0

def TSEB_2T(Tc,Ts,Ta_K,u,ea,p,Rn_sw_veg, Rn_sw_soil, Rn_lw_veg, Rn_lw_soil,LAI, 
            hc,z_0M, d_0, zu, zt, 
            leaf_width=0.1,z0_soil=0.01, alpha_PT=1.26,f_c=1,CalcG=[1,0.35]):
    ''' TSEB using component canopy and soil temperatures.

    Calculates the turbulent fluxes by the Two Source Energy Balance model 
    using canopy and soil component temperatures that were derived or measured
    previously.
    
    Parameters
    ----------
    Ts : float
        Soil Temperature (Kelvin).
    Tc : float
        Canopy Temperature (Kelvin).
    Ta_K : float 
        Air temperature (Kelvin).
    u : float 
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Rn_sw_veg : float
        Canopy net shortwave radiation (W m-2).
    Rn_sw_soil : float
        Soil net shortwave radiation (W m-2).
    Rn_lw_veg : float
        Canopy net longwave radiation (W m-2).
    Rn_lw_soil : float
        Soil net longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    hc : float
        Canopy height (m).
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    zu : float
        Height of measurement of windspeed (m).
    zt : float
        Height of measurement of air temperature (m).
    leaf_width : float, optional
        average/effective leaf width (m).
    z0_soil : float, optional
        bare soil aerodynamic roughness length (m).
    alpha_PT : float, optional
        Priestley Taylor coeffient for canopy potential transpiration, 
        use 1.26 by default.
    CalcG : tuple(int,list), optional
        Method to calculate soil heat flux,parameters.
        
            * (1,G_ratio): default, estimate G as a ratio of Rn_soil, default Gratio=0.35.
            * (0,G_constant) : Use a constant G, usually use 0 to ignore the computation of G.
            * (2,G_param) : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.CalcG_TimeDiff`).
    
    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
    LE_C : float
        Canopy latent heat flux (W m-2).
    H_C : float
        Canopy sensible heat flux (W m-2).
    LE_S : float
        Soil latent heat flux (W m-2).
    H_S : float
        Soil sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_s : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_a : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    n_iterations : int
        number of iterations until convergence of L.
        
    References
    ----------
    .. [Kustas1997] Kustas, W. P., and J. M. Norman (1997), A two-source approach for estimating
        turbulent fluxes using multiple angle thermal infrared observations,
        Water Resour. Res., 33(6), 1495-1508,
        http://dx.doi.org/10.1029/97WR00704.
    '''
    
    from math import exp
    #Air density (kg m-3)
    rho_a=met.CalcRho(p, ea, Ta_K)
    #Heat capacity or air at constant pressure (273.15K) (J kg-1 K-1)
    Cp=met.CalcC_p(p, ea)
    #Compute Net Radiation
    Rn_soil=Rn_sw_soil+Rn_lw_soil
    Rn_veg=Rn_sw_veg+Rn_lw_veg
    Rn=Rn_soil+Rn_veg
    #Compute Soil Heat Flux Ratio
    if CalcG[0]==0:
        G=CalcG[1]
    elif CalcG[0]==1:
        G=CalcG_Ratio(Rn_soil, CalcG[1])
    elif CalcG[0]==2:
        G=CalcG_TimeDiff (Rn_soil, CalcG[1])
    # Slope of the saturation Vapour pressure (kPa) (Allen 1998 A3.9)
    Ta=Ta_K-273.15
    delta=2504*exp(17.27*Ta/(Ta+237.3))/((Ta+237.3)**2.)
    #Calculate roughness for heat transport
    z_0H=res.CalcZ_0H(z_0M,kB=kB)
    # EStimate real LAI
    F=LAI/f_c
    #Define Variables for iteration
    L=float('inf')
    # Initial values to start iteration
    L_old=1.0
    u_old=1e36
    #First Guess Friction Velocity 
    max_iterations=ITERATIONS
    u_friction=MO.CalcU_star (u, zu, L, d_0,z_0M)
    u_friction =max(u_friction_min, u_friction)
    #Start the Monin-Obukhov iteration
    for n_iterations in range(max_iterations):
        # Check whether flux constrains applied
        flag=0
        # Resistances
        U_C=MO.CalcU_C (u_friction, hc, d_0, z_0M)
        u_S=MO.CalcU_Goudriaan (U_C, hc, LAI, leaf_width, z0_soil)
        u_d_zm = MO.CalcU_Goudriaan (U_C, hc, LAI, leaf_width,d_0+z_0M)
        R_x=res.CalcR_X_Norman(F, leaf_width, u_d_zm)
        R_s=res.CalcR_S_Kustas(u_S, Ts-Tc)
        # Calculate Aerodynamic Resistace
        R_a=res.CalcR_A ( zt,  u_friction, L, d_0, z_0H)
        R_s=max( 1e-3,R_s)
        R_x=max( 1e-3,R_x)
        R_a=max( 1e-3,R_a)
        # Compute air temperature at the canopy interface
        T_ac=((Ta_K/R_a)+(Ts/R_s)+(Tc/R_x))/((1/R_a)+(1/R_s)+(1/R_x))
        T_ac=max( 1e-3,T_ac)
        #Canopy Sensible heat flux (Norman et al 1995)
        H_c=rho_a*Cp*(Tc-T_ac)/R_x
        #Assume no condensation in the canopy (LE_c<0)
        if H_c > Rn_veg:
            H_c=Rn_veg
            flag=1
        #Assume no thermal inversion in the canopy
        if H_c < Rn_veg*(1-alpha_PT*delta/(delta+psicr)) and Rn_veg > 0:
        #if H_c < 0:
            H_c=0
            flag=2
        #Soil Sensible heat flux (Norman et al 1995)
        H_s=rho_a*Cp*(Ts-T_ac)/R_s
        #Assume that there is no condensation in the soil (LE_s<0)
        if H_s > Rn_soil-G and (Rn_soil-G) > 0:
            H_s=Rn_soil-G
            flag=3
        #Assume no thermal inversion in the soil
        if H_s < 0 and Rn_soil-G > 0:
        #if H_s < 0:
            H_s=0
            flag=4                                 
        #Evaporation Rate (Kustas and Norman 1999)
        H=H_s+H_c
        LE=(Rn-G-H)
        #Monin-Obukhov Lenght
        L=MO.CalcL (u_friction, Ta_K, rho_a, Cp, H, LE)
        # Calculate again the friction velocity with the new stability correctios        
        u_friction=MO.CalcU_star (u, zu, L, d_0,z_0M)
        #Difference of Heat Flux between interations
        L_diff=abs(L-L_old)/abs(L_old)
        L_old=L
        if abs(L_old)==0.0: L_old=1e-36
        # Calculate the change in friction velocity
        u_diff=abs(u_friction-u_old)/abs(u_old)
        u_old=u_friction
        #Avoid very low friction velocity values
        u_friction =max(u_friction_min, u_friction)
        #Stop the iteration if differences are below the threshold
        if L_diff < L_thres and u_diff < u_thres:
            break
    # END of iteration
    #Compute soil and canopy heat fluxes
    LE_s=Rn_soil-G-H_s
    LE_c=Rn_veg-H_c
    return flag,T_ac,LE_c,H_c,LE_s,H_s,G,R_s,R_x,R_a,u_friction, L,n_iterations

def  TSEB_PT(Tr_K,vza,Ta_K,u,ea,p,Sdn_dir, Sdn_dif, fvis,fnir,sza,Lsky,
            LAI,hc,emisVeg,emisGrd,spectraVeg,spectraGrd,z_0M,d_0,zu,zt,
            leaf_width=0.1,z0_soil=0.01,alpha_PT=1.26,f_c=1.0,f_g=1.0,wc=1.0,
            CalcG=[1,0.35]):
    '''Priestley-Taylor TSEB

    Calculates the Priestley Taylor TSEB fluxes using a single observation of
    composite radiometric temperature and using resistances in series.
    
    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    Ta_K : float 
        Air temperature (Kelvin).
    u : float 
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sdn_dir : float
        Beam solar irradiance (W m-2).
    Sdn_dif : float
        Difuse solar irradiance (W m-2).
    fvis : float
        Fraction of shortwave radiation corresponding to the PAR region (400-700nm).
    fnir : float
        Fraction of shortwave radiation corresponding to the NIR region (700-2500nm).
    sza : float
        Solar Zenith Angle (degrees).
    Lsky : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    hc : float
        Canopy height (m).
    emisVeg : float
        Leaf emissivity.
    emisGrd : flaot
        Soil emissivity.
    spectraVeg : dict('rho_leaf_vis', 'tau_leaf_vis', 'rho_leaf_nir','tau_leaf_nir')
        Leaf spectrum dictionary.        

            rho_leaf_vis : float
                leaf bihemispherical reflectance in the visible (400-700 nm).
            tau_leaf_vis : float
                leaf bihemispherical transmittance in the visible (400-700nm).
            rho_leaf_nir : float
                leaf bihemispherical reflectance in the optical infrared (700-2500nm),
            tau_leaf_nir : float
                leaf bihemispherical transmittance in the optical  infrared (700-2500nm).
    spectraGrd : dict('rho rsoilv', 'rsoiln')
        Soil spectrum dictonary.
        
            rsoilv : float
                soil bihemispherical reflectance in the visible (400-700 nm).
            rsoiln : float
                soil bihemispherical reflectance in the optical infrared (700-2500nm).
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    zu : float
        Height of measurement of windspeed (m).
    zt : float
        Height of measurement of air temperature (m).
    leaf_width : float, optional
        average/effective leaf width (m).
    z0_soil : float, optional
        bare soil aerodynamic roughness length (m).
    alpha_PT : float, optional
        Priestley Taylor coeffient for canopy potential transpiration, 
        use 1.26 by default.
    x_LAD : float, optional
        Campbell 1990 leaf inclination distribution function chi parameter.
    f_c : float, optional
        Fractional cover.
    f_g : float, optional
        Fraction of vegetation that is green.
    wc : float, optional
        Canopy width to height ratio.
    CalcG : tuple(int,list), optional
        Method to calculate soil heat flux,parameters.
        
            * (1,G_ratio): default, estimate G as a ratio of Rn_soil, default Gratio=0.35.
            * (0,G_constant) : Use a constant G, usually use 0 to ignore the computation of G.
            * (2,G_param) : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.CalcG_TimeDiff`).
    
    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    Ts : float
        Soil temperature  (Kelvin).
    Tc : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
    S_nS : float
        Soil net shortwave radiation (W m-2)
    S_nC : float
        Canopy net shortwave radiation (W m-2)
    L_nS : float
        Soil net longwave radiation (W m-2)
    L_nC : float
        Canopy net longwave radiation (W m-2)
    LE_C : float
        Canopy latent heat flux (W m-2).
    H_C : float
        Canopy sensible heat flux (W m-2).
    LE_S : float
        Soil latent heat flux (W m-2).
    H_S : float
        Soil sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_s : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_a : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    n_iterations : int
        number of iterations until convergence of L.

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29,
        http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''
    
    # Create the output variables
    [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,
     u_friction, L,counter]=[0 for i in range(19)]
    #Define Variables for iteration
    L=float('inf')
    # Initial values to start iteration
    L_old=1
    u_old=1e36
    Tc= 0.0
    if LAI==0: # One Source Energy Balance
        z_0M=z0_soil
        d_0=5*z_0M
        spectraGrd=fvis*spectraGrd['rsoilv']+fnir* spectraGrd['rsoiln']
        [flag,S_nS, L_nS, LE_S,H_S,G,R_a,u_friction, L,counter]=OSEB(Tr_K,
            Ta_K,u,ea,p,Sdn_dir+ Sdn_dif,Lsky,emisGrd,spectraGrd,z_0M,d_0,zu,zt,CalcG=CalcG)
        return [flag, Tr_K, Tc, Ta_K,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                R_s,R_x,R_a,u_friction, L,counter]
    # calculate the general parameters
    #Real LAI and clumping index
    F=LAI/f_c
    omega0=CI.CalcOmega0_Kustas(LAI, f_c,isLAIeff=True)
    Omega=CI.CalcOmega_Kustas(omega0,sza,wc=wc)
    f_theta = CalcFthetaCampbell(vza, F, wc=wc,Omega0=omega0)   #Fraction of vegetation observed by the sensor
    rho= met.CalcRho(p, ea, Ta_K)  #Air density
    c_p = met.CalcC_p(p, ea)  #Heat capacity of air  
    # initial ustar
    max_iterations=ITERATIONS
    u_friction = MO.CalcU_star(u, zu, L, d_0,z_0M)
    z_0H=res.CalcZ_0H(z_0M,kB=kB)
    # Net shortwave radiation
    LAI_eff=F*Omega
    S_nC, S_nS = rad.CalcSnCampbell (LAI_eff, sza, Sdn_dir, Sdn_dif, fvis,
                 fnir, spectraVeg['rho_leaf_vis'], spectraVeg['tau_leaf_vis'],
                spectraVeg['rho_leaf_nir'], spectraVeg['tau_leaf_nir'], 
                spectraGrd['rsoilv'], spectraGrd['rsoiln'])    

    # First assume that canpy temperature equals the minumum of Air or radiometric T
    Tc=min(Tr_K, Ta_K)
    flag,Ts=CalcT_S(Tr_K, Tc, f_theta)
    if flag ==255:
        return [flag, Tr_K, Tc, Ta_K,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                R_s,R_x,R_a,u_friction, L,counter]
    
    # loop for estimating stability, stop when difference in consecutives L is below 0.01
    for n_iterations in range(max_iterations):
        flag=0
        # Calculate net longwave radiation with current values of Tc and Ts
        L_nC, L_nS = rad.CalcLnKustas (Tc, Ts, Lsky, LAI,emisVeg, emisGrd)
        delta_R_n = L_nC + S_nC
        R_n_soil=S_nS+L_nS
        # calculate the aerodynamic resistances
        R_a=res.CalcR_A ( zt, u_friction, L, d_0, z_0H)
        # Calculate wind speed at the canopy height
        U_C=MO.CalcU_C (u_friction, hc, d_0, z_0M)
        # Calculate soil and canopy resistances
        u_S=MO.CalcU_Goudriaan (U_C, hc, LAI, leaf_width, z0_soil)
        u_d_zm = MO.CalcU_Goudriaan (U_C, hc, LAI, leaf_width,d_0+z_0M)
        R_x=res.CalcR_X_Norman(F, leaf_width, u_d_zm)
        R_s=res.CalcR_S_Kustas(u_S, Ts-Ta_K)
        R_s=max( 1e-3,R_s)
        R_x=max( 1e-3,R_x)
        R_a=max( 1e-3,R_a)
        # calculate the canopy and soil temperatures using the Priestley Taylor appoach
        # first we assume potential canopy transpiration
        H_C = CalcH_C_PT(delta_R_n, f_g, Ta_K, p, c_p, alpha_PT)
        Tc= CalcT_C_Series(Tr_K,Ta_K, R_a, R_x, R_s, f_theta, H_C, rho, c_p)
        # get soil temperature in Kelvin
        flag,Ts = CalcT_S(Tr_K, Tc, f_theta)
        if flag ==255:
            return [flag, Tr_K, Tc, Ta_K,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                    R_s,R_x,R_a,u_friction, L,counter]
        R_s=res.CalcR_S_Kustas(u_S, Ts-Ta_K)
        R_s=max( 1e-3,R_s)
        # get air temperature at canopy interface in Celsius
        T_AC = (( Ta_K/R_a + Ts/R_s + Tc/R_x )
            /(1.0/R_a + 1.0/R_s + 1.0/R_x))
        # calculate soil fluxes
        H_S =  rho * c_p * (Ts - T_AC)/ R_s
        #Compute Soil Heat Flux Ratio
        if CalcG[0]==0:
            G=CalcG[1]
        elif CalcG[0]==1:
            G=CalcG_Ratio(R_n_soil, CalcG[1])
        elif CalcG[0]==2:
            G=CalcG_TimeDiff (R_n_soil, CalcG[1])
        LE_S = R_n_soil - G - H_S
        LE_C = delta_R_n - H_C        
        # Check daytime soil latent heat fluxes
        if LE_S < 0: #and R_n_soil > 0 and delta_R_n > 0:
            LE_S=0.0
            H_S=R_n_soil-G
            Ts,T_AC=CalcT_S_Series(Tr_K,Ta_K,R_a,R_x,R_s,f_theta,H_S,rho,c_p)
            #R_s=CalcR_S_Kustas(u_S, Ts-Ta_K)
            #T_AC = (( Ta_K/R_a + Ts/R_s + Tc/R_x )/(1.0/R_a + 1.0/R_a + 1.0/R_x))
            flag,Tc=CalcT_C(Tr_K, Ts, f_theta)
            if flag ==255:
               return [flag, Tr_K, Tc, Ta_K,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                       R_s,R_x,R_a,u_friction, L,counter]

            H_C=rho * c_p * (Tc - T_AC)/ R_x
            LE_C=delta_R_n-H_C
            flag=3
            # Check for daytime canopy latent heat fluxes
            if LE_C < 0:
                LE_C=0
                H_C=delta_R_n
                # Use parallel version to avoid iteration
                Tc=CalcT_C_Series(Tr_K,Ta_K, R_a, R_x, R_s, f_theta, H_C, rho, c_p)
                flag,Ts=CalcT_S(Tr_K, Tc, f_theta)
                if flag ==255:
                    return [flag, Tr_K, Tc, Ta_K,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                            R_s,R_x,R_a,u_friction, L,counter]
                R_s=res.CalcR_S_Kustas(u_S, Ts-Ta_K)
                R_s=max( 1e-3,R_s)
                T_AC = (( Ta_K/R_a + Ts/R_s + Tc/R_x )/(1.0/R_a + 1.0/R_s + 1.0/R_x))
                H_S=rho*c_p*(Ts-T_AC)/(R_s)
                G=R_n_soil-H_S
                flag=5
        # calculate total fluxes
        H = H_C + H_S
        LE = LE_C + LE_S
        #Monin-Obukhov Lenght
        L=MO.CalcL (u_friction, Ta_K, rho, c_p, H, LE)
        #Difference of Heat Flux between interations
        L_diff=abs(L-L_old)/abs(L_old)
        L_old=L
        if abs(L_old)==0: L_old=1e-36
        # Calculate again the friction velocity with the new stability correctios        
        u_friction=MO.CalcU_star (u, zu, L, d_0,z_0M)
        # Calculate the change in friction velocity
        u_diff=abs(u_friction-u_old)/abs(u_old)
        u_old=u_friction
        #Avoid very low friction velocity values
        u_friction =max(u_friction_min, u_friction)
        #Stop the iteration if differences are below the threshold
        if L_diff < L_thres and u_diff < u_thres:
            break
        
    return flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,R_s,R_x,R_a,u_friction, L,n_iterations
    
def  DTD(Tr_K_0,Tr_K_1,vza,Ta_K_0,Ta_K_1,u,ea,p,Sdn_dir,Sdn_dif, fvis,fnir,sza,
             Lsky,LAI,hc,emisVeg,emisGrd,spectraVeg,spectraGrd,z_0M,d_0,zu,zt,
             leaf_width=0.1,z0_soil=0.01,alpha_PT=1.26,f_c=1.0,f_g=1.0,wc=1.0,
             CalcG=[1,0.35]):
    ''' Calculate daytime Dual Time Difference TSEB fluxes
    
    Parameters
    ----------
    Tr_K_0 : float
        Radiometric composite temperature around sunrise(Kelvin).
    Tr_K_1 : float
        Radiometric composite temperature near noon (Kelvin).
    vza : float
        View Zenith Angle near noon (degrees).
    Ta_K_0 : float 
        Air temperature around sunrise (Kelvin).
    Ta_K_1 : float 
        Air temperature near noon (Kelvin).
    u : float 
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sdn_dir : float
        Beam solar irradiance (W m-2).
    Sdn_dif : float
        Difuse solar irradiance (W m-2).
    fvis : float
        Fraction of shortwave radiation corresponding to the PAR region (400-700nm).
    fnir : float
        Fraction of shortwave radiation corresponding to the NIR region (700-2500nm).
    sza : float
        Solar zenith angle (degrees).
    Lsky : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    hc : float
        Canopy height (m).
    emisVeg : float
        Leaf emissivity.
    emisGrd : flaot
        Soil emissivity.
    spectraVeg : dict('rho_leaf_vis', 'tau_leaf_vis', 'rho_leaf_nir','tau_leaf_nir')
        Leaf spectrum dictionary.        
            
            rho_leaf_vis : float
                leaf bihemispherical reflectance in the visible (400-700 nm).
            tau_leaf_vis : float
                leaf bihemispherical transmittance in the visible (400-700nm).
            rho_leaf_nir : float
                leaf bihemispherical reflectance in the optical infrared (700-2500nm),
            tau_leaf_nir : float
                leaf bihemispherical transmittance in the optical  infrared (700-2500nm).
    spectraGrd : dict('rho rsoilv', 'rsoiln')
        Soil spectrum dictonary.
        
            rsoilv : float
                soil bihemispherical reflectance in the visible (400-700 nm).
            rsoiln : float
                soil bihemispherical reflectance in the optical infrared (700-2500nm).
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    zu : float
        Height of measurement of windspeed (m).
    zt : float
        Height of measurement of air temperature (m).
    leaf_width : Optional[float]
        average/effective leaf width (m).
    z0_soil : Optional[float]
        bare soil aerodynamic roughness length (m).
    alpha_PT : Optional[float]
        Priestley Taylor coeffient for canopy potential transpiration, 
        use 1.26 by default.
    x_LAD : Optional[float]
        Campbell 1990 leaf inclination distribution function chi parameter.
    f_c : Optiona;[float]
        Fractional cover.
    f_g : Optional[float]
        Fraction of vegetation that is green.
    wc : Optional[float]
        Canopy width to height ratio.
    CalcG : Optional[tuple(int,list)]
        Method to calculate soil heat flux,parameters.
        
            * (1,G_ratio): default, estimate G as a ratio of Rn_soil, default Gratio=0.35.
            * (0,G_constant) : Use a constant G, usually use 0 to ignore the computation of G.
            * (2,G_param) : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.CalcG_TimeDiff`).
    
    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    Ts : float
        Soil temperature  (Kelvin).
    Tc : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
    S_nS : float
        Soil net shortwave radiation (W m-2).
    S_nC : float
        Canopy net shortwave radiation (W m-2).
    L_nS : float
        Soil net longwave radiation (W m-2).
    L_nC : float
        Canopy net longwave radiation (W m-2).
    LE_C : float
        Canopy latent heat flux (W m-2).
    H_C : float
        Canopy sensible heat flux (W m-2).
    LE_S : float
        Soil latent heat flux (W m-2).
    H_S : float
        Soil sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_s : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_a : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    Ri : float
        Richardson number.
    n_iterations : int
        number of iterations until convergence of L.

    References
    ----------
    .. [Norman2000] Norman, J. M., W. P. Kustas, J. H. Prueger, and G. R. Diak (2000),
        Surface flux estimation using radiometric temperature: A dual-temperature-difference
        method to minimize measurement errors, Water Resour. Res., 36(8), 2263-2274,
        http://dx.doi.org/10.1029/2000WR900033.
    .. [Guzinski2015] Guzinski, R., Nieto, H., Stisen, S., and Fensholt, R. (2015) Inter-comparison
        of energy balance and hydrological models for land surface energy flux estimation over
        a whole river catchment, Hydrol. Earth Syst. Sci., 19, 2017-2036,
        http://dx.doi.org/10.5194/hess-19-2017-2015.
    '''
    
    # Create the output variables
    [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
         R_s,R_x,R_a,u_friction,L,Ri,count]=[0 for i in range(20)]
    
    # Calculate the general parameters
    rho= met.CalcRho(p, ea, Ta_K_1)  # Air density
    c_p = met.CalcC_p(p, ea)  # Heat capacity of air 
    omega0 = CI.CalcOmega0_Kustas(LAI, f_c, isLAIeff=True) # Clumping factor at nadir
    omega = CI.CalcOmega_Kustas(omega0,sza,wc=wc) # Clumping factor at an angle    
    F = LAI/f_c # Real LAI    
    f_theta = CalcFthetaCampbell(vza, F, wc=wc,Omega0=omega0)   # Fraction of vegetation observed by the sensor
    
    # Calculate the Richardson number    
    Ri = MO.CalcRichardson (u, zu, d_0,Tr_K_0, Tr_K_1, Ta_K_0, Ta_K_1)
    # L is not used in the DTD, since Richardson number is used instead to
    # avoid dependance on non-differential temperatures. But it is still saved
    # in the output for testing purposes.    
    L = 0
        
    # Calculate the soil resistance
    # First calcualte u_S, wind speed at the soil surface
    u_friction = MO.CalcU_star(u, zu, Ri, d_0,z_0M, useRi=True)    
    u_C = MO.CalcU_C(u_friction, hc, d_0, z_0M)
    u_S=MO.CalcU_Goudriaan (u_C, hc, LAI, leaf_width, z0_soil)
    # deltaT based on equation from Guzinski et. al., 2015    
    deltaT=(Tr_K_1 - Tr_K_0) - (Ta_K_1- Ta_K_0)    
    R_s=res.CalcR_S_Kustas(u_S, deltaT)
    
    # Calculate the other resistances resistances
    z_0H=res.CalcZ_0H(z_0M,kB=kB)
    R_a=res.CalcR_A (zu, u_friction, Ri, d_0, z_0H, useRi=True)
    u_d_zm = MO.CalcU_Goudriaan (u_C, hc, LAI, leaf_width,d_0+z_0M)
    R_x=res.CalcR_X_Norman(F, leaf_width, u_d_zm)
    
    # Calcualte short wave net radiation of canopy and soil
    LAI_eff=F*omega
    S_nC, S_nS = rad.CalcSnCampbell (LAI_eff, sza, Sdn_dir,Sdn_dif, 
           fvis,fnir, spectraVeg['rho_leaf_vis'], spectraVeg['tau_leaf_vis'],
            spectraVeg['rho_leaf_nir'], spectraVeg['tau_leaf_nir'], 
            spectraGrd['rsoilv'], spectraGrd['rsoiln'])    
    
    # First assume that canpy temperature equals the minumum of Air or radiometric T
    Tc=min(Tr_K_1, Ta_K_1)
    flag,Ts=CalcT_S(Tr_K_1, Tc, f_theta)
    if flag ==255:
        return [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                R_s,R_x,R_a,u_friction, L,Ri,count]
    
    # Outer loop until canopy and soil temperatures have stabilised 
    Tc_prev = 0
    iteration = 0
    while abs(Tc - Tc_prev) > 0.1 and iteration < ITERATIONS:
        flag = 0
        Tc_prev = Tc
        iteration += 1

        # Inner loop to iterativelly reduce alpha_PT in case latent heat flux 
        # from the soil is negative
        LE_S = -1
        alpha_PT_rec = alpha_PT + 0.1  
        while LE_S < 0:
            
            # There cannot be negative transpiration from the vegetation
            alpha_PT_rec -= 0.1
            if alpha_PT_rec <= 0:
                alpha_PT_rec = 0 
                flag = 5
            elif alpha_PT_rec < alpha_PT:
                flag = 3
                
            # Calculate net longwave radiation with current values of Tc and Ts
            L_nC, L_nS = rad.CalcLnKustas (Tc, Ts, Lsky, LAI_eff,emisVeg, emisGrd)
            
            # Calculate total net radiation of soil and canopy        
            delta_R_n = L_nC + S_nC
            R_n_soil=S_nS+L_nS
            
            # Calculate sensible heat fluxes at time t1
            H_C= CalcH_C_PT(delta_R_n, f_g, Ta_K_1, p, c_p, alpha_PT_rec)
            H = CalcH_DTD_series(Tr_K_1, Tr_K_0, Ta_K_1, Ta_K_0, rho, c_p, f_theta,
                R_s, R_a, R_x, H_C)
            H_S = H - H_C
                               
            # Calculate ground heat flux
            if CalcG[0]==0:
                G=CalcG[1]
            elif CalcG[0]==1:
                G=CalcG_Ratio(R_n_soil, CalcG[1])
            elif CalcG[0]==2:
                G=CalcG_TimeDiff (R_n_soil, CalcG[1])
            
            # Calculate latent heat fluxes as residuals
            LE_C = delta_R_n - H_C
            LE_S = R_n_soil - H_S - G

            # Special case if there is no transpiration from vegetation. 
            # In that case, there should also be no evaporation from the soil
            # and sensible heat flux of the soil should depend only on the soil
            # energy budget          
            if LE_C == 0:              
                H_S = R_n_soil - G
                LE_S = 0
    
            # Recalculate soil and canopy temperatures. They are used only for  
            # estimation of longwave radiation, so the use of non-differential Tr
            # and Ta shouldn't affect the turbulent fluxes much
            Tc = CalcT_C_Series(Tr_K_1,Ta_K_1, R_a, R_x, R_s, f_theta, H_C, rho, c_p)
            flag_t,Ts = CalcT_S (Tr_K_1, Tc, f_theta)
            if flag_t ==255:
                return [flag_t, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                    R_s,R_x,R_a,u_friction, L,Ri,count]
    
    # L is only calculated for testing purposes
    L=MO.CalcL (u_friction, Ta_K_1, rho, c_p, H, LE_C + LE_S)               
    return [flag, Ts, Tc, T_AC,S_nS, S_nC, L_nS,L_nC, LE_C,H_C,LE_S,H_S,G,
                R_s,R_x,R_a,u_friction, L,Ri,count]        

def  OSEB(Tr_K,Ta_K,u,ea,p,Sdn,Lsky,emis,albedo,z_0M,d_0,zu,zt, CalcG=[1,0.35]):
    '''Calulates bulk fluxes from a One Source Energy Balance model

    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    Ta_K : float 
        Air temperature (Kelvin).
    u : float 
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sdn : float
        Solar irradiance (W m-2).
    Lsky : float
        Downwelling longwave radiation (W m-2)
    emis : float
        Surface emissivity.
    albedo : float
        Surface broadband albedo.        
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    zu : float
        Height of measurement of windspeed (m).
    zt : float
        Height of measurement of air temperature (m).
    CalcG : Optional[tuple(int,list)]
        Method to calculate soil heat flux,parameters.
        
            * (1,G_ratio): default, estimate G as a ratio of Rn_soil, default Gratio=0.35
            * (0,G_constant) : Use a constant G, usually use 0 to ignore the computation of G
            * (2,G_param) : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.CalcG_TimeDiff`).
    
    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    S_n : float
        Net shortwave radiation (W m-2)
    L_n : float
        Net longwave radiation (W m-2)
    LE : float
        Latent heat flux (W m-2).
    H : float
        Sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_a : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    n_iterations : int
        number of iterations until convergence of L.
    '''
   
    # initially stable conditions
    #Define Variables for iteration
    L=float('inf')
    # Initial values to start iteration
    L_old=1
    u_old=1e36
    flag=0
    # calculate the general parameters
    rho= met.CalcRho(p, ea, Ta_K)  #Air density
    c_p = met.CalcC_p(p, ea)  #Heat capacity of air
    max_iterations=ITERATIONS
    u_friction = MO.CalcU_star(u, zu, L, d_0,z_0M)
    z_0H=res.CalcZ_0H(z_0M,kB=kB)
    # Net radiation
    S_n,L_n=rad.CalcRnOSEB(Sdn,Lsky, Tr_K, emis, albedo)
    R_n=S_n+L_n
    #Compute Soil Heat Flux
    if CalcG[0]==0:
        G_calc=CalcG[1]
    elif CalcG[0]==1:
        G_calc=CalcG_Ratio(R_n, CalcG[1])
    elif CalcG[0]==2:
        G_calc=CalcG_TimeDiff (R_n, CalcG[1])
    # loop for estimating stability, stop when difference in consecutives L is below 0.01
    for n_iterations in range(max_iterations):
        G=G_calc
        # calculate the aerodynamic resistances
        R_a=res.CalcR_A ( zt, u_friction, L, d_0, z_0H)
        R_a=max( 1e-3,R_a)
        # calculate bulk fluxes
        H =  rho * c_p * (Tr_K - Ta_K)/ R_a
        LE = R_n -G - H
        # Avoid negative ET during daytime
        if LE<0:
            LE=0
            G=R_n-G-LE
        # now L can be recalculated
        L=MO.CalcL (u_friction, Ta_K, rho, c_p, H, LE)
        #Difference of Heat Flux between interations
        L_diff=abs(L-L_old)/abs(L_old)
        L_old=L
        if abs(L_old)==0: L_old=1e-36
        # Calculate again the friction velocity with the new stability correctios        
        u_friction=MO.CalcU_star (u, zu, L, d_0,z_0M)
        # Calculate the change in friction velocity
        u_diff=abs(u_friction-u_old)/abs(u_old)
        u_old=u_friction
        #Avoid very low friction velocity values
        u_friction =max(u_friction_min, u_friction)
        #Stop the iteration if differences are below the threshold
        if L_diff < L_thres and u_diff < u_thres:
            break
    
    return flag,S_n, L_n, LE,H,G,R_a,u_friction, L,n_iterations
  
def CalcFthetaCampbell(theta,F,wc=1,Omega0=1, x_LAD=1):
    '''Calculates the fraction of vegetatinon observed at an angle.
    
    Parameters
    ----------
    theta : float
        Angle of incidence (degrees).
    F : float
        Real Leaf (Plant) Area Index.
    wc : float
        Ratio of vegetation height versus width, optional (default = 1).
    Omega0 : float
        Clumping index at nadir, optional (default =1).
    x_LAD : float
        Chi parameter for the ellipsoidal Leaf Angle Distribution function, 
        use x_LAD=1 for a spherical LAD.
    
    Returns
    -------
    f_theta : float
        fraction of vegetation obsserved at an angle.
    
    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    from math import radians, exp
    # First calcualte the angular clumping factor Omega based on eq (3) from
    # W.P. Kustas, J.M. Norman,  Agricultural and Forest Meteorology 94 (1999)
    OmegaTheta = Omega0 / (Omega0 + (1.0 - Omega0) * exp(-2.2 * radians(theta)**(3.8 - 0.46 * wc)))    #CHECK: should theta here be in degrees or radians
    # Estimate the beam extinction coefficient based on a elipsoidal LAD function
    # Eq. 15.4 of Campbell and Norman (1998)
    K_be=rad.CalcKbe_Campbell(theta,x_LAD)
    ftheta=1.0-exp(-K_be*OmegaTheta*F)
    return ftheta

def CalcG_TimeDiff (R_n, G_param=[12.0,0.35, 3.0,24.0]):
    ''' Estimates Soil Heat Flux as function of time and net radiation.
    
    Parameters
    ----------
    R_n : float
        Net radiation (W m-2).
    G_param : tuple(float,float,float,float)
        tuple with parameters required (time, Amplitude,phase_shift,shape).
        
            time: float 
                time of interest (decimal hours).
            Amplitude : float 
                maximum value of G/Rn, amplitude, default=0.35.
            phase_shift : float
                shift of peak G relative to solar noon (default 3hrs after noon).
            shape : float
                shape of G/Rn, default 24 hrs.
    
    Returns
    -------
    G : float
        Soil heat flux (W m-2).

    References
    ----------
    .. [Santanello2003] Joseph A. Santanello Jr. and Mark A. Friedl, 2003: Diurnal Covariation in
        Soil Heat Flux and Net Radiation. J. Appl. Meteor., 42, 851-862,
        http://dx.doi.org/10.1175/1520-0450(2003)042<0851:DCISHF>2.0.CO;2.'''
    
    from math import cos, pi
    # Get parameters
    time=12.0-G_param[0]
    A = G_param[1]
    phase_shift=G_param[2]
    B = G_param[3]
    G_ratio=A*cos(2.0*pi*(time+phase_shift)/B)
    G = R_n * G_ratio
    return G

def CalcG_Ratio(Rn_soil,G_ratio=0.35):
    '''Estimates Soil Heat Flux as ratio of net soil radiation.
    
    Parameters
    ----------
    Rn_soil : float
        Net soil radiation (W m-2).
    G_ratio : float, optional
        G/Rn_soil ratio, default=0.35.
    
    Returns
    -------
    G : float
        Soil heat flux (W m-2).

    References
    ----------
    .. [Choudhury1987] B.J. Choudhury, S.B. Idso, R.J. Reginato, Analysis of an empirical model
        for soil heat flux under a growing wheat crop for estimating evaporation by an
        infrared-temperature based energy balance equation, Agricultural and Forest Meteorology,
        Volume 39, Issue 4, 1987, Pages 283-297,
        http://dx.doi.org/10.1016/0168-1923(87)90021-9.
    '''

    G= G_ratio*Rn_soil
    return G

def CalcH_C (T_C, T_A, R_A, rho, c_p):
    '''Calculates canopy sensible heat flux in a parallel resistance network.
    
    Parameters
    ----------
    T_C : float
        Canopy temperature (K).
    T_A : float
        Air temperature (K).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    
    Returns
    -------
    H_C : float
        Canopy sensible heat flux (W m-2).'''

    H_C = rho*c_p*(T_C-T_A)/R_A
    return H_C

def  CalcH_C_PT (delta_R_ni, f_g, T_a_K, P, c_p, alpha):
    '''Calculates canopy sensible heat flux based on the Priestley and Taylor formula.
    
    Parameters
    ----------
    delta_R_ni : float
        net radiation divergence of the vegetative canopy (W m-2).
    f_g : float
        fraction of vegetative canopy that is green.
    T_a_K : float
        air temperature (Kelvin).
    P : float
        air pressure (mb).
    c_p : float
        heat capacity of moist air (J kg-1 K-1).
    alpha : float 
        the Priestley Taylor parameter.
    
    Returns
    -------
    H_C : float
        Canopy sensible heat flux (W m-2).

    References
    ----------
    Equation 14 in [Norman1995]_
    '''  

    # slope of the saturation pressure curve (kPa./deg C)
    s = met.CalcDeltaVaporPressure( T_a_K)
    s=s*10 # to mb
    # latent heat of vaporisation (MJ./kg)
    Lambda=met.CalcLambda(T_a_K)
    # psychrometric constant (mb C-1)
    gama=met.CalcPsicr(P,Lambda)
    s_gama = s / (s + gama)
    H_C = delta_R_ni * (1.0 - alpha * f_g * s_gama)
    return H_C

def CalcH_DTD_parallel (T_R1, T_R0, T_A1, T_A0, rho, c_p, f_theta1, R_S1, R_A1, R_AC1, H_C1):
    '''Calculates the DTD total sensible heat flux at time 1 with resistances in parallel.
    
    Parameters
    ----------
    T_R1 : float
        radiometric surface temperature at time t1 (K).
    T_R0 : float
        radiometric surface temperature at time t0 (K).
    T_A1 : float
        air temperature at time t1 (K).
    T_A0 : float
        air temperature at time t0 (K).
    rho : float
        air density at time t1 (kg m-3).
    cp : float
        heat capacity of moist air (J kg-1 K-1).
    f_theta_1 : float
        fraction of radiometer field of view that is occupied by vegetative cover at time t1.
    R_S1 : float
        resistance to heat transport from the soil surface at time t1 (s m-1).
    R_A1 : float
        resistance to heat transport in the surface layer at time t1 (s m-1).
    R_A1 : float
        resistance to heat transport at the canopy interface at time t1 (s m-1).
    H_C1 : float
        canopy sensible heat flux at time t1 (W m-2).
    
    Returns
    -------
    H : float
        Total sensible heat flux at time t1 (W m-2).

    References
    ----------
    .. [Guzinski2013] Guzinski, R., Anderson, M. C., Kustas, W. P., Nieto, H., and Sandholt, I. (2013)
        Using a thermal-based two source energy balance model with time-differencing to
        estimate surface energy fluxes with day-night MODIS observations,
        Hydrol. Earth Syst. Sci., 17, 2809-2825,
        http://dx.doi.org/10.5194/hess-17-2809-2013.
    '''


    #% Ignore night fluxes
    H = (rho*c_p *(((T_R1-T_R0)-(T_A1-T_A0))/((1.0-f_theta1)*(R_A1+R_S1))) +
        H_C1*(1.0-((f_theta1*R_AC1)/((1.0-f_theta1)*(R_A1+R_S1)))))
    return H   
    
def CalcH_DTD_series(T_R1, T_R0, T_A1, T_A0, rho, c_p, f_theta, R_S, R_A, R_x, H_C):
    '''Calculates the DTD total sensible heat flux at time 1 with resistances in series
    
    Parameters
    ----------
    T_R1 : float
        radiometric surface temperature at time t1 (K).
    T_R0 : float
        radiometric surface temperature at time t0 (K).
    T_A1 : float
        air temperature at time t1 (K).
    T_A0 : float
        air temperature at time t0 (K).
    rho : float
        air density at time t1 (kg m-3).
    cp : float
        heat capacity of moist air (J kg-1 K-1).
    f_theta : float
        fraction of radiometer field of view that is occupied by vegetative cover at time t1.
    R_S : float
        resistance to heat transport from the soil surface at time t1 (s m-1).
    R_A : float
        resistance to heat transport in the surface layer at time t1 (s m-1).
    R_x : float
        Canopy boundary resistance to heat transport at time t1 (s m-1).
    H_C : float
        canopy sensible heat flux at time t1 (W m-2).
    
    Returns
    -------
    H : float
        Total sensible heat flux at time t1 (W m-2).

    References
    ----------
    .. [Guzinski2014] Guzinski, R., Nieto, H., Jensen, R., and Mendiguren, G. (2014)
        Remotely sensed land-surface energy fluxes at sub-field scale in heterogeneous
        agricultural landscape and coniferous plantation, Biogeosciences, 11, 5021-5046,
        http://dx.doi.org/10.5194/bg-11-5021-2014.
    '''
    H = rho*c_p*((T_R1-T_R0)-(T_A1-T_A0))/((1.0-f_theta)*R_S + R_A) + \
        H_C*((1.0-f_theta)*R_S - f_theta*R_x)/((1.0-f_theta)*R_S + R_A)
    return H
     
def CalcH_S (T_S, T_A, R_A, R_S, rho, c_p):
    '''Calculates soil sensible heat flux in a parallel resistance network.
    
    Parameters
    ----------
    T_S : float
        Soil temperature (K).
    T_A : float
        Air temperature (K).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    R_A : float
        Aerodynamic resistance at the soil boundary layer (s m-1).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
   
    Returns
    -------
    H_C : float
        Canopy sensible heat flux (W m-2).

    References
    ----------
    Equation 7 in [Norman1995]_
    '''

    H_S = rho*c_p*((T_S-T_A)/(R_S+R_A))
    return H_S
    
def  CalcT_C (T_R, T_S, f_theta):
    '''Estimates canopy temperature from the directional composite radiometric temperature.
    
    Parameters
    ----------
    T_R : float
        Directional Radiometric Temperature (K).
    T_S : float
        Soil Temperature (K).
    f_theta : float
        Fraction of vegetation observed.

    Returns
    -------
    flag : int
        Error flag if inversion not possible (255).
    T_C : float
        Canopy temperature (K).

    References
    ----------
    Eq. 1 in [Norman1995]_
    '''
    
    if ( T_R**4 - (1.0 - f_theta)*T_S**4 ) >0:
        T_C = ( ( T_R**4 - (1.0 - f_theta)*T_S**4 ) /f_theta)**0.25
        flag=0
    else:
        flag=255        
        T_C=1e-6
    return [flag,T_C]


def CalcT_C_Series(Tr_K,Ta_K, R_a, R_x, R_s, f_theta, H_C, rho, c_p):
    '''Estimates canopy temperature from canopy sensible heat flux and 
    resistance network in series.
    
    Parameters
    ----------
    Tr_K : float
        Directional Radiometric Temperature (K).
    Ta_K : float
        Air Temperature (K).
    R_a : float
        Aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk aerodynamic resistance to heat transport at the canopy boundary layer (s m-1).
    R_s : float
        Aerodynamic resistance to heat transport at the soil boundary layer (s m-1).
    f_theta : float
        Fraction of vegetation observed.
    H_C : float
        Sensible heat flux of the canopy (W m-2).
    rho : float
        Density of air (km m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    
    Returns
    -------
    T_c : float
        Canopy temperature (K).
    
    References
    ----------
    Eqs. A5-A13 in [Norman1995]_'''
    
    T_R_K_4=Tr_K**4
    # equation A7 from Norman 1995, linear approximation of temperature of the canopy
    T_C_lin = (( Ta_K/R_a + Tr_K/(R_s*(1.0-f_theta)) 
        + H_C*R_x/(rho*c_p)*(1.0/R_a + 1.0/R_s + 1.0/R_x)) 
        /(1.0/R_a + 1.0/R_s + f_theta/(R_s*(1.0 - f_theta))))
    # equation A12 from Norman 1995
    T_D = (T_C_lin*(1+R_s/R_a) - H_C*R_x/(rho*c_p)*(1.0 + R_s/R_x + R_s/R_a)
            - Ta_K*R_s/R_a)
    # equation A11 from Norman 1995
    delta_T_C = ((T_R_K_4 - f_theta*T_C_lin**4 - (1.0-f_theta)*T_D**4) 
        / (4.0* (1.0-f_theta)* T_D**3* (1.0 + R_s/R_a) + 4.0*f_theta*T_C_lin**3))
    # get canopy temperature in Kelvin
    Tc = T_C_lin + delta_T_C
    return Tc
   
def CalcT_CS_Norman (F, vza_n, vza_f, T_n, T_f,wc=1,x_LAD=1, omega0=1):
    '''Estimates canopy and soil temperature by analytical inversion of Eq 1 in [Norman1995]
    of two directional radiometric observations. Ignoring shawows.
    
    Parameters
    ----------
    F : float
        Real Leaf (Plant) Area Index.
    vza_n : float
        View Zenith Angle during the nadir observation (degrees).
    vza_f : float
        View Zenith Angle during the oblique observation (degrees).
    T_n : float
        Radiometric temperature in the nadir obsevation (K).
    T_f : float
        Radiometric temperature in the oblique observation (K).
    wc : float,optional
        Canopy height to width ratio, use wc=1 by default.
    x_LAD : float,optional
        Chi parameter for the ellipsoildal Leaf Angle Distribution function of 
        Campbell 1988 [default=1, spherical LIDF].
    omega0 : float,optional
        Clumping index at nadir, use omega0=1 by default.
    
    Returns
    -------
    Tc : float
        Canopy temperature (K).
    Ts : float
        Soil temperature (K).
    
    References
    ----------
    inversion of Eq. 1 in [Norman1995]_
    '''


    # Calculate the fraction of vegetation observed by each angle
    f_theta_n=CalcFthetaCampbell(vza_n, F, wc=wc,Omega0=omega0,x_LAD=x_LAD)
    f_theta_f=CalcFthetaCampbell(vza_f, F, wc=wc,Omega0=omega0,x_LAD=x_LAD)
    # Solve the sytem of two unknowns and two equations
    Ts_4=(f_theta_f*T_n**4-f_theta_n*T_f**4)/(f_theta_f-f_theta_n)
    Tc_4=(T_n**4-(1.0-f_theta_n)*Ts_4)/f_theta_n
    if Tc_4< 0 or Ts_4 <0:
        return float('nan'),float('nan')
    Tc_K=Tc_4**0.25
    Ts_K=Ts_4**0.25
    return Tc_K, Ts_K

def  CalcT_S (T_R, T_C, f_theta):
    '''Estimates soil temperature from the directional LST.
    
    Parameters
    ----------
    T_R : float
        Directional Radiometric Temperature (K).
    T_C : float
        Canopy Temperature (K).
    f_theta : float
        Fraction of vegetation observed.

    Returns
    -------
    flag : float
        Error flag if inversion not possible (255).
    T_S: float
        Soil temperature (K).
    
    References
    ----------
    Eq. 1 in [Norman1995]_'''

    if ( T_R**4 - f_theta*T_C**4 ) >0:
        T_S = ( ( T_R**4 - f_theta*T_C**4) /(1.0 - f_theta))**0.25
        flag=0
    else:
        flag=255
        T_S=1e-6
    return [flag,T_S]

def CalcT_S_Series(Tr_K,Ta_K,R_a,R_x,R_s,f_theta,H_S,rho,c_p):
    '''Estimates soil temperature from soil sensible heat flux and 
    resistance network in series.
    
    Parameters
    ----------
    Tr_K : float
        Directional Radiometric Temperature (K).
    Ta_K : float
        Air Temperature (K).
    R_a : float
        Aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk aerodynamic resistance to heat transport at the canopy boundary layer (s m-1).
    R_s : float
        Aerodynamic resistance to heat transport at the soil boundary layer (s m-1).
    f_theta : float
        Fraction of vegetation observed.
    H_S : float
        Sensible heat flux of the soil (W m-2).
    rho : float
        Density of air (km m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
        
    Returns
    -------
    T_s: float
        Soil temperature (K).
    T_c : float
        Air temperature at the canopy interface (K).
    
    References
    ----------
    Eqs. A15-A19 from [Norman1995]_'''

    #Eq. A.15 Norman 1995
    T_ac_lin=(((Ta_K/R_a)+(Tr_K/(f_theta*R_x))-
        (((1.0-f_theta)/(f_theta*R_x))*H_S*R_s/(rho*c_p))+H_S/(rho*c_p))/
        ((1.0/R_a)+(1.0/R_x)+(1.0-f_theta)/(f_theta*R_x)))    
    #Eq. A.17 Norman 1995
    T_e=T_ac_lin*(1.0+(R_x/R_a))-H_S*R_x/(rho*c_p)-Ta_K*R_x/R_a    
     #Eq. A.16 Norman 1995
    Delta_T_ac=((Tr_K**4-(1.0-f_theta)*(H_S*R_s/(rho*c_p)+T_ac_lin)**4-f_theta*T_e**4)/
        (4*f_theta*T_e**3.0*(1.0+(R_x/R_a))+4.0*(1.0-f_theta)*(H_S*R_s/(rho*c_p)+T_ac_lin)**3))
    #Eq. A.18 Norman 1995
    T_ac=T_ac_lin+Delta_T_ac    
    T_s=T_ac+H_S*R_s/(rho*c_p)
    return [T_s,T_ac]
