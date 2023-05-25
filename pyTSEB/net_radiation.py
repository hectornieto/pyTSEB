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
@author: Hector Nieto (hector.nieto@ica.csic.es)

DESCRIPTION
===========
This package contains functions for estimating the net shortwave and longwave radiation
for soil and canopy layers. Additional packages needed are.

* :doc:`meteo_utils` for the estimation of meteorological variables.

PACKAGE CONTENTS
================
* :func:`calc_difuse_ratio` estimation of fraction of difuse shortwave radiation.
* :func:`calc_emiss_atm` Atmospheric emissivity.
* :func:`calc_K_be_Campbell` Beam extinction coefficient.
* :func:`calc_L_n_Kustas` Net longwave radiation for soil and canopy layers.
* :func:`calc_Sn_Campbell` Net shortwave radiation.
* :func:`calc_tau_below_Campbell` Radiation transmission through a canopy.
'''

import numpy as np

from . import meteo_utils as met

#==============================================================================
# List of constants used in the netRadiation Module
#==============================================================================
# Stephan Boltzmann constant (W m-2 K-4)
SB = 5.670373e-8
TAUD_STEP_SIZE_DEG = 5


def _calc_taud(x_lad, lai):

    taud = 0
    for angle in range(0, 90, TAUD_STEP_SIZE_DEG):
        angle = np.radians(angle)
        akd = calc_K_be_Campbell(angle, x_lad, radians=True)
        taub = np.exp(-akd * lai)
        taud += taub * np.cos(angle) * np.sin(angle) * np.radians(TAUD_STEP_SIZE_DEG)

    return 2.0 * taud


def calc_difuse_ratio(S_dn, sza, press=1013.25, SOLAR_CONSTANT=1320):
    """Fraction of difuse shortwave radiation.

    Partitions the incoming solar radiation into PAR and non-PR and
    diffuse and direct beam component of the solar spectrum.

    Parameters
    ----------
    S_dn : float
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
    """

    # Convert input scalars to numpy arrays
    S_dn, sza, press = map(np.asarray, (S_dn, sza, press))
    difvis, difnir, fvis, fnir = [np.zeros(S_dn.shape) for i in range(4)]
    fvis = fvis + 0.6
    fnir = fnir + 0.4

    # Calculate potential (clear-sky) visible and NIR solar components
    # Weiss & Norman 1985
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = calc_potential_irradiance_weiss(
        sza, press=press, SOLAR_CONSTANT=SOLAR_CONSTANT)

    # Potential total solar radiation
    potvis = np.asarray(Rdirvis + Rdifvis)
    potvis[potvis <= 0] = 1e-6
    potnir = np.asarray(Rdirnir + Rdifnir)
    potnir[potnir <= 0] = 1e-6
    fclear = S_dn / (potvis + potnir)
    fclear = np.minimum(1.0, fclear)

    # Partition S_dn into VIS and NIR
    fvis = potvis / (potvis + potnir)  # Eq. 7
    fnir = potnir / (potvis + potnir)  # Eq. 8
    fvis = np.clip(fvis, 0.0, 1.0)
    fnir = 1.0 - fvis

    # Estimate direct beam and diffuse fractions in VIS and NIR wavebands
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.9] = 0.9
    dirvis = (Rdirvis / potvis) * (1. - ((.9 - ratiox) / .7)**.6667)  # Eq. 11
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.88] = 0.88
    dirnir = (Rdirnir / potnir) * \
        (1. - ((.88 - ratiox) / .68)**.6667)  # Eq. 12

    dirvis = np.clip(dirvis, 0.0, 1.0)
    dirnir = np.clip(dirnir, 0.0, 1.0)
    difvis = 1.0 - dirvis
    difnir = 1.0 - dirnir

    return np.asarray(difvis), np.asarray(
        difnir), np.asarray(fvis), np.asarray(fnir)


def calc_emiss_atm(ea, t_a_k):
    '''Atmospheric emissivity

    Estimates the effective atmospheric emissivity for clear sky.

    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
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

    emiss_air = 1.24 * (ea / t_a_k)**(1. / 7.)  # Eq. 11 in [Brutsaert1975]_

    return np.asarray(emiss_air)


def calc_longwave_irradiance(ea, t_a_k, p=1013.25, z_T=2.0, h_C=2.0):
    '''Longwave irradiance

    Estimates longwave atmospheric irradiance from clear sky.
    By default there is no lapse rate correction unless air temperature
    measurement height is considerably different than canopy height, (e.g. when
    using NWP gridded meteo data at blending height)

    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (K).
    p : float
        air pressure (mb)
    z_T: float
        air temperature measurement height (m), default 2 m.
    h_C: float
        canopy height (m), default 2 m,

    Returns
    -------
    L_dn : float
        Longwave atmospheric irradiance (W m-2) above the canopy
    '''

    lapse_rate = met.calc_lapse_rate_moist(t_a_k, ea, p)
    t_a_surface = t_a_k - lapse_rate * (h_C - z_T)
    emisAtm = calc_emiss_atm(ea, t_a_surface)
    L_dn = emisAtm * met.calc_stephan_boltzmann(t_a_surface)
    return np.asarray(L_dn)


def calc_K_be_Campbell(theta, x_lad=1, radians=False):
    ''' Beam extinction coefficient

    Calculates the beam extinction coefficient based on [Campbell1998]_ ellipsoidal
    leaf inclination distribution function.

    Parameters
    ----------
    theta : float
        incidence zenith angle.
    x_lad : float, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_lad=1 for a spherical LAD.
    radians : bool, optional
        Should be True if theta is in radians.
        Default is False.

    Returns
    -------
    K_be : float
        beam extinction coefficient.
    x_lad: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_lad=1 for a spherical LAD.

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    '''

    if not radians:
        theta = np.radians(theta)

    K_be = (np.sqrt(x_lad**2 + np.tan(theta)**2)
            / (x_lad + 1.774 * (x_lad + 1.182)**-0.733))

    return K_be


def calc_L_n_Kustas(T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD=1):
    ''' Net longwave radiation for soil and canopy layers

    Estimates the net longwave radiation for soil and canopy layers unisg based on equation 2a
    from [Kustas1999]_ and incorporated the effect of the Leaf Angle Distribution based on
    [Campbell1998]_

    Parameters
    ----------
    T_C : float
        Canopy temperature (K).
    T_S : float
        Soil temperature (K).
    L_dn : float
        Downwelling atmospheric longwave radiation (w m-2).
    lai : float
        Effective Leaf (Plant) Area Index.
    emisVeg : float
        Broadband emissivity of vegetation cover.
    emisGrd : float
        Broadband emissivity of soil.
    x_lad: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_lad=1 for a spherical LAD.

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

    # Get the diffuse transmitance
    _, _, _, taudl = calc_spectra_Cambpell(lai,
                                          np.zeros(emisVeg.shape),
                                          1.0 - emisVeg,
                                          np.zeros(emisVeg.shape),
                                          1.0 - emisGrd,
                                          x_lad=x_LAD,
                                          lai_eff=None)

    # calculate long wave emissions from canopy, soil and sky
    L_C = emisVeg * met.calc_stephan_boltzmann(T_C)
    L_S = emisGrd * met.calc_stephan_boltzmann(T_S)

    # calculate net longwave radiation divergence of the soil
    L_nS = taudl * L_dn + (1.0 - taudl) * L_C - L_S
    L_nC = (1.0 - taudl) * (L_dn + L_S - 2.0 * L_C)
    return np.asarray(L_nC), np.asarray(L_nS)


def calc_L_n_Campbell(T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD=1):
    ''' Net longwave radiation for soil and canopy layers

    Estimates the net longwave radiation for soil and canopy layers unisg based on equation 2a
    from [Kustas1999]_ and incorporated the effect of the Leaf Angle Distribution based on [Campbell1998]_

    Parameters
    ----------
    T_C : float
        Canopy temperature (K).
    T_S : float
        Soil temperature (K).
    L_dn : float
        Downwelling atmospheric longwave radiation (w m-2).
    lai : float
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

    # calculate long wave emissions from canopy, soil and sky
    L_C = emisVeg * met.calc_stephan_boltzmann(T_C)
    L_C[np.isnan(L_C)] = 0
    L_S = emisGrd * met.calc_stephan_boltzmann(T_S)
    L_S[np.isnan(L_S)] = 0
    # Calculate the canopy spectral properties
    _, albl, _, taudl = calc_spectra_Cambpell(lai,
                                              np.zeros(emisVeg.shape),
                                              1.0 - emisVeg,
                                              np.zeros(emisVeg.shape),
                                              1.0 - emisGrd,
                                              x_lad=x_LAD,
                                              lai_eff=None)

    # calculate net longwave radiation divergence of the soil
    L_nS = emisGrd * taudl * L_dn + emisGrd * (1.0 - taudl) * L_C - L_S
    L_nC = (1 - albl) * (1.0 - taudl) * (L_dn + L_S) - 2.0 * (1.0 - taudl) * L_C
    L_nC[np.isnan(L_nC)] = 0
    L_nS[np.isnan(L_nS)] = 0
    return np.asarray(L_nC), np.asarray(L_nS)


def calc_potential_irradiance_weiss(
        sza,
        press=1013.25,
        SOLAR_CONSTANT=1320,
        fnir_ini=0.5455):
    ''' Estimates the potential visible and NIR irradiance at the surface

    Parameters
    ----------
    sza : float
        Solar Zenith Angle (degrees)
    press : Optional[float]
        atmospheric pressure (mb)

    Returns
    -------
    Rdirvis : float
        Potential direct visible irradiance at the surface (W m-2)
    Rdifvis : float
        Potential diffuse visible irradiance at the surface (W m-2)
    Rdirnir : float
        Potential direct NIR irradiance at the surface (W m-2)
    Rdifnir : float
        Potential diffuse NIR irradiance at the surface (W m-2)

    based on Weiss & Normat 1985, following same strategy in Cupid's RADIN4 subroutine.
    '''

    # Convert input scalars to numpy arrays
    sza, press = map(np.asarray, (sza, press))

    # Set defaout ouput values
    Rdirvis, Rdifvis, Rdirnir, Rdifnir, w = [
        np.zeros(sza.shape) for i in range(5)]

    coszen = np.cos(np.radians(sza))
    # Calculate potential (clear-sky) visible and NIR solar components
    # Weiss & Norman 1985
    # Correct for curvature of atmos in airmas (Kasten and Young,1989)
    i = sza < 90
    airmas = 1.0 / coszen
    # Visible PAR/NIR direct beam radiation
    Sco_vis = SOLAR_CONSTANT * (1.0 - fnir_ini)
    Sco_nir = SOLAR_CONSTANT * fnir_ini
    # Directional trasnmissivity
    # Calculate water vapour absorbance (Wang et al 1976)
    # A=10**(-1.195+.4459*np.log10(1)-.0345*np.log10(1)**2)
    # opticalDepth=np.log(10.)*A
    # T=np.exp(-opticalDepth/coszen)
    # Asssume that most absortion of WV is at the NIR
    Rdirvis[i] = (Sco_vis * np.exp(-.185 * (press[i] / 1313.25) * airmas[i])
                  - w[i]) * coszen[i]  # Modified Eq1 assuming water vapor absorption
    # Rdirvis=(Sco_vis*exp(-.185*(press/1313.25)*airmas))*coszen
    # #Eq. 1
    Rdirvis = np.maximum(0, Rdirvis)
    # Potential diffuse radiation
    # Eq 3                                      #Eq. 3
    Rdifvis[i] = 0.4 * (Sco_vis * coszen[i] - Rdirvis[i])
    Rdifvis = np.maximum(0, Rdifvis)

    # Same for NIR
    # w=SOLAR_CONSTANT*(1.0-T)
    w = SOLAR_CONSTANT * \
        10**(-1.195 + .4459 * np.log10(coszen[i]) - .0345 * np.log10(coszen[i])**2)  # Eq. .6
    Rdirnir[i] = (Sco_nir * np.exp(-0.06 * (press[i] / 1313.25)
                                   * airmas[i]) - w) * coszen[i]  # Eq. 4
    Rdirnir = np.maximum(0, Rdirnir)

    # Potential diffuse radiation
    Rdifnir[i] = 0.6 * (Sco_nir * coszen[i] - Rdirvis[i] - w)  # Eq. 5
    Rdifnir = np.maximum(0, Rdifnir)
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = map(
        np.asarray, (Rdirvis, Rdifvis, Rdirnir, Rdifnir))
    return Rdirvis, Rdifvis, Rdirnir, Rdifnir

def calc_spectra_Cambpell(lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad=1, lai_eff=None):
    """ Canopy spectra

    Estimate canopy spectral using the [Campbell1998]_
    Radiative Transfer Model

    Parameters
    ----------
    lai : float
        Effective Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float
        Soil bihemispherical reflectance
    x_lad : float,  optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    lai_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.

    Returns
    -------
    albb : float or array_like
        Beam (black sky) canopy albedo
    albd : float or array_like
        Diffuse (white sky) canopy albedo
    taubt : float or array_like
        Beam (black sky) canopy transmittance
    taudt : float or array_like
        Beam (white sky) canopy transmittance

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    """

    # calculate aborprtivity
    amean = 1.0 - rho_leaf - tau_leaf
    amean_sqrt = np.sqrt(amean)
    del rho_leaf, tau_leaf, amean

    # Calculate canopy beam extinction coefficient
    # Modification to include other LADs
    if lai_eff is None:
        lai_eff = np.asarray(lai)
    else:
        lai_eff = np.asarray(lai_eff)

    # D I F F U S E   C O M P O N E N T S
    # Integrate to get the diffuse transmitance
    taud = _calc_taud(x_lad, lai)

    # Diffuse light canopy reflection coefficients  for a deep canopy
    akd = -np.log(taud) / lai
    rcpy= (1.0 - amean_sqrt) / (1.0 + amean_sqrt)  # Eq 15.7
    rdcpy = 2.0 * akd * rcpy / (akd + 1.0)  # Eq 15.8

    # Diffuse canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akd * lai
    del akd
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    xnum = (rdcpy * rdcpy - 1.0) * neg_exp
    xden = (rdcpy * rho_soil - 1.0) + rdcpy * (rdcpy - rho_soil) * d_neg_exp
    taudt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rdcpy - rho_soil) / (rdcpy * rho_soil - 1.0)) * d_neg_exp
    albd = (rdcpy + fact) / (1.0 + rdcpy * fact)  # Eq 15.9
    del rdcpy, fact

    # B E A M   C O M P O N E N T S
    # Direct beam extinction coeff (spher. LAD)
    akb = calc_K_be_Campbell(sza, x_lad)  # Eq. 15.4

    # Direct beam canopy reflection coefficients for a deep canopy
    rbcpy = 2.0 * akb * rcpy / (akb + 1.0)  # Eq 15.8
    del rcpy, sza, x_lad
    # Beam canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akb * lai_eff
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    del amean_sqrt, akb, lai_eff
    xnum = (rbcpy * rbcpy - 1.0) * neg_exp
    xden = (rbcpy * rho_soil - 1.0) + rbcpy * (rbcpy - rho_soil) * d_neg_exp
    taubt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rbcpy - rho_soil) / (rbcpy * rho_soil - 1.0)) * d_neg_exp
    del expfac
    albb = (rbcpy + fact) / (1.0 + rbcpy * fact)  # Eq 15.9
    del rbcpy, fact

    taubt, taudt, albb, albd, rho_soil = map(np.array,
                                             [taubt, taudt, albb, albd, rho_soil])

    taubt[np.isnan(taubt)] = 1
    taudt[np.isnan(taudt)] = 1
    albb[np.isnan(albb)] = rho_soil[np.isnan(albb)]
    albd[np.isnan(albd)] = rho_soil[np.isnan(albd)]

    return albb, albd, taubt, taudt


def calc_Sn_Campbell(lai, sza, S_dn_dir, S_dn_dif, fvis, fnir, rho_leaf_vis,
                     tau_leaf_vis, rho_leaf_nir, tau_leaf_nir, rsoilv, rsoiln,
                     x_LAD=1, LAI_eff=None):
    ''' Net shortwave radiation

    Estimate net shorwave radiation for soil and canopy below a canopy using the [Campbell1998]_
    Radiative Transfer Model, and implemented in [Kustas1999]_

    Parameters
    ----------
    lai : float
        Effecive Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    S_dn_dir : float
        Broadband incoming beam shortwave radiation (W m-2).
    S_dn_dif : float
        Broadband incoming diffuse shortwave radiation (W m-2).
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
    x_lad : float, optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    LAI_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.

    Returns
    -------
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
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

    rho_leaf = np.array((rho_leaf_vis, rho_leaf_nir))
    tau_leaf = np.array((tau_leaf_vis, tau_leaf_nir))
    rho_soil = np.array((rsoilv, rsoiln))
    albb, albd, taubt, taudt = calc_spectra_Cambpell(lai,
                                                     sza,
                                                     rho_leaf,
                                                     tau_leaf,
                                                     rho_soil,
                                                     x_lad=x_LAD,
                                                     lai_eff=LAI_eff)

    Sn_C = ((1.0 - taubt[0]) * (1.0- albb[0]) * S_dn_dir*fvis
            + (1.0 - taubt[1]) * (1.0- albb[1]) * S_dn_dir*fnir
            + (1.0 - taudt[0]) * (1.0- albd[0]) * S_dn_dif*fvis
            + (1.0 - taudt[1]) * (1.0- albd[1]) * S_dn_dif*fnir)
            
    Sn_S = (taubt[0] * (1.0 - rsoilv) * S_dn_dir*fvis
            + taubt[1] * (1.0 - rsoiln) * S_dn_dir*fnir
            + taudt[0] * (1.0 - rsoilv) * S_dn_dif*fvis
            + taudt[1] * (1.0 - rsoiln) * S_dn_dif*fnir)
    
    return np.asarray(Sn_C), np.asarray(Sn_S)


def leafangle_2_chi(alpha):
    alpha = np.radians(alpha)
    x_lad = ((alpha / 9.65) ** (1. / -1.65)) - 3.

    return x_lad


def chi_2_leafangle(x_lad):
    alpha = 9.65 * (3. + x_lad) ** -1.65
    alpha = np.degrees(alpha)
    return alpha

