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
@author: Hector Nieto (hector.nieto@ica.csic.es)

DESCRIPTION
===========
This package contains the main routines inherent of Two Source Energy Balance `TSEB` models.
Additional functions needed in TSEB, such as computing of net radiation or estimating the
resistances to heat and momentum transport are imported.

* :doc:`net_radiation` for the estimation of net radiation and radiation partitioning.
* :doc:`clumping_index` for the estimatio of canopy clumping index.
* :doc:`meteo_utils` for the estimation of meteorological variables.
* :doc:`resistances` for the estimation of the resistances to heat and momentum transport.
* :doc:`MO_similarity` for the estimation of the Monin-Obukhov length and MOST-related variables.
* :doc:`wind_profile` for the estimation of wind attenuation profile

PACKAGE CONTENTS
================

TSEB models
-----------
* :func:`TSEB_2T` TSEB using derived/measured canopy and soil component temperatures.
* :func:`TSEB_PT` Priestley-Taylor TSEB using a
                  single observation of composite radiometric temperature.
* :func:`DTD` Dual-Time Differenced TSEB using composite radiometric temperatures at two times:
              early morning and near afternoon.

OSEB models
-----------
* :func:`OSEB`. One Source Energy Balance Model.

Ancillary functions
-------------------
* :func:`calc_F_theta_campbell`. Gap fraction estimation.
* :func:`calc_G_time_diff`. Santanello & Friedl (2003) [Santanello2003]_ soil heat flux model.
* :func:`calc_G_ratio`. Soil heat flux as a fixed fraction of net radiation [Choudhury1987]_.
* :func:`calc_H_C`. canopy sensible heat flux in a parallel resistance network.
* :func:`calc_H_C_PT`. Priestley- Taylor Canopy sensible heat flux.
* :func:`calc_H_DTD_parallel`. Priestley- Taylor Canopy sensible
                               heat flux for DTD and resistances in parallel.
* :func:`calc_H_DTD_series`. Priestley- Taylor Canopy sensible heat flux
                             for DTD and resistances in series.
* :func:`calc_H_S`. Soil heat flux with resistances in parallel.
* :func:`calc_T_C`. Canopy temperature form composite radiometric temperature.
* :func:`calc_T_C_series.` Canopy temperature from canopy sensible
                           heat flux and resistance in series.
* :func:`calc_T_CS_Norman`. Component temperatures from dual angle
                            composite radiometric temperatures.
* :func:`calc_T_CS_4SAIL`. Component temperatures from dual angle composite radiometric tempertures.
                           Using 4SAIl for the inversion.
* :func:`calc_4SAIL_emission_param`. Effective surface reflectance, and emissivities for soil and
                                     canopy using 4SAIL.
* :func:`calc_T_S`. Soil temperature from form composite radiometric temperature.
* :func:`calc_T_S_series`. Soil temperature from soil sensible heat flux and resistance in series.
'''

from collections import deque
import time

import numpy as np
from pypro4sail.four_sail import foursail

from . import meteo_utils as met
from . import resistances as res
from . import MO_similarity as MO
from . import net_radiation as rad
from . import clumping_index as CI
from . import wind_profile as wnd
from . import energy_combination_ET as pet

# ==============================================================================
# List of constants used in TSEB model and sub-routines
# ==============================================================================
# Threshold for relative change in Monin-Obukhov lengh to stop the iterations
L_thres = 0.001
# mimimun allowed friction velocity
U_FRICTION_MIN = 0.01
U_S_MIN = 0.01
U_C_MIN = 0.01
R_A_MIN = 1e-1
R_A_MAX = None
RES_MIN = 1e-1
RES_MAX = None

# Maximum number of interations
ITERATIONS = 15
# kB coefficient
KB_1_DEFAULT = 0.0
# Stephan Boltzmann constant (W m-2 K-4)
SB = 5.670373e-8

# Resistance formulation constants
KUSTAS_NORMAN_1999 = 0
CHOUDHURY_MONTEITH_1988 = 1
MCNAUGHTON_VANDERHURK = 2
CHOUDHURY_MONTEITH_ALPHA_1988 = 3
HADHIGHI_AND_OR_2015 = 4

# Soil heat flux formulation constants
G_CONSTANT = 0
G_RATIO = 1
G_TIME_DIFF = 2
G_TIME_DIFF_SIGMOID = 3

# Flag constants
F_ALL_FLUXES = 0  # All fluxes produced with no reduction of PT parameter (i.e. positive soil evaporation)
F_ZERO_LE_C = 1  # Negative canopy latent heat flux, forced to zero
F_ZERO_H_C = 2  # Negative canopy sensible heat flux, forced to zero
F_ZERO_LE_S = 3  # Negative soil evaporation, forced to zero (the PT parameter is reduced in TSEB-PT and DTD)
F_ZERO_H_S = 4  # Negative soil sensible heat flux, forced to zero
F_ZERO_LE = 5  # No positive latent fluxes found, G recomputed to close the energy balance (G=Rn-H)
F_ALL_FLUXES_OS = 10  # All positive fluxes for soil only, produced using one-source energy balance (OSEB) model.
F_ZERO_LE_OS = 15  # No positive latent fluxes found using OSEB, G recomputed to close the energy balance (G=Rn-H)
F_INVALID = 255  # Arithmetic error. BAD data, it should be discarded

# Steps for decreasing transpiration efficiency in TSEB-SW
STEP_BETA = 0.05
# Steps for increasing soil surface resistance to water transport in TSEB-SW
STEP_RSS = 500.
MAX_RST = 5000.
RELATIVE_INCREASE = 0.10
STEP_RST = 10.
# Steps for increasing surface resistance to water transport in TSEB-PM
MAX_RC = 5000.
STEP_RC = 5.

def TSEB_2T(T_C,
            T_S,
            T_A_K,
            u,
            ea,
            p,
            Sn_C,
            Sn_S,
            L_dn,
            LAI,
            h_C,
            emis_C,
            emis_S,
            z_0M,
            d_0,
            z_u,
            z_T,
            leaf_width=0.1,
            z0_soil=0.01,
            alpha_PT=1.26,
            x_LAD=1.0,
            f_c=1.0,
            f_g=1.0,
            w_C=1.0,
            resistance_form=None,
            calcG_params=None,
            const_L=None,
            kB=KB_1_DEFAULT,
            massman_profile=None,
            verbose=True):
    """ TSEB using component canopy and soil temperatures.

    Calculates the turbulent fluxes by the Two Source Energy Balance model
    using canopy and soil component temperatures that were derived or measured
    previously.

    Parameters
    ----------
    T_S : float
        Soil Temperature (Kelvin).
    T_C : float
        Canopy Temperature (Kelvin).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2)
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    h_C : float
        Canopy height (m).
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
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
    w_C : float, optional
        Canopy width to height ratio.

    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.

            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.

    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with
                                                       G_param list of parameters
                                                       (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.

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
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
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
    """

    if resistance_form is None:
        resistance_form = [0, {}]
    if calcG_params is None:
        calcG_params = [[1], 0.35]
    if massman_profile is None:
        massman_profile = [0, []]
    # Convert float scalars into numpy arrays and check parameters size
    T_C = np.asarray(T_C)
    [T_S,
     T_A_K,
     u,
     ea,
     p,
     Sn_C,
     Sn_S,
     L_dn,
     LAI,
     h_C,
     emis_C,
     emis_S,
     z_0M,
     d_0,
     z_u,
     z_T,
     leaf_width,
     z0_soil,
     alpha_PT,
     x_LAD,
     f_c,
     f_g,
     w_C,
     calcG_array] = map(_check_default_parameter_size,
                        [T_S,
                         T_A_K,
                         u,
                         ea,
                         p,
                         Sn_C,
                         Sn_S,
                         L_dn,
                         LAI,
                         h_C,
                         emis_C,
                         emis_S,
                         z_0M,
                         d_0,
                         z_u,
                         z_T,
                         leaf_width,
                         z0_soil,
                         alpha_PT,
                         x_LAD,
                         f_c,
                         f_g,
                         w_C,
                         calcG_params[1]],
                        [T_C] * 24)

    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    # calcG_params[1] = None
    # Create the output variables
    [flag, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x,
        R_A, iterations] = [np.zeros(T_S.shape, np.float32)+np.nan for i in range(12)]
    T_AC = T_A_K.copy()

    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(T_S.shape) + np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(T_S.shape) * const_L)
        max_iterations = 1  # No iteration

    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air
    z_0H = res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport

    # Calculate LAI dependent parameters for dataset where LAI > 0
    F = np.asarray(LAI / f_c)  # Real LAI
    # Calculate LAI dependent parameters for dataset where LAI > 0
    omega0 = CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)

    # And the net longwave radiation
    Ln_C, Ln_S = rad.calc_L_n_Campbell(T_C, T_S, L_dn, LAI, emis_C, emis_S, x_LAD=x_LAD)

    # Compute Net Radiation
    Rn_S = Sn_S + Ln_S
    Rn_C = Sn_C + Ln_C
    Rn = Rn_S + Rn_C

    # Compute Soil Heat Flux
    i = np.ones(Rn_S.shape, dtype=bool)
    G[i] = calc_G([calcG_params[0], calcG_array], Rn_S, i)
    # iteration of the Monin-Obukhov length
    u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(U_FRICTION_MIN, u_friction))
    l_queue = deque([np.array(L)], 6)
    l_converged = np.asarray(np.zeros(T_S.shape)).astype(bool)
    l_diff_max = np.inf

    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        if np.all(l_converged[i]):
            if verbose:
                if l_converged[i].size == 0:
                    print("Finished iterations with no valid solution")
                else:
                    print(f"Finished interations with a max. L diff: {l_diff_max}")
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        if verbose:
            print("Iteration: %d, non-converged pixels: %d, max L diff: %f, total time: %f, loop time: %f" %
                  (n_iterations, np.sum(~l_converged[i]), l_diff_max, total_duration, loop_duration))
        iterations[np.logical_and(~l_converged, flag != F_INVALID)] = n_iterations

        i = np.logical_and(~l_converged, flag != F_INVALID)
        iterations[i] = n_iterations
        flag[i] = F_ALL_FLUXES

        # Calculate aerodynamic resistances
        R_A[i], R_x[i], R_S[i] = calc_resistances(
                resistance_form, {"R_A": {"z_T": z_T[i], "u_friction": u_friction[i], "L": L[i],
                                          "d_0": d_0[i], "z_0H": z_0H[i]},
                                  "R_x": {"u_friction": u_friction[i], "h_C": h_C[i],
                                          "d_0": d_0[i],
                                          "z_0M": z_0M[i], "L": L[i], "F": F[i], "LAI": LAI[i],
                                          "leaf_width": leaf_width[i],
                                          "z0_soil": z0_soil[i],
                                          "massman_profile": massman_profile,
                                          "res_params": {k: res_params[k][i] for k in res_params.keys()}},
                                  "R_S": {"u_friction": u_friction[i], "h_C": h_C[i],
                                          "d_0": d_0[i],
                                          "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i],
                                          "LAI": LAI[i], "leaf_width": leaf_width[i],
                                          "z0_soil": z0_soil[i], "z_u": z_u[i],
                                          "deltaT": T_S[i] - T_AC[i], 'u': u[i], 'rho': rho[i],
                                          "c_p": c_p[i], "f_cover": f_c[i], "w_C": w_C[i],
                                          "massman_profile": massman_profile,
                                          "res_params": {k: res_params[k][i] for k in res_params.keys()}}
                                   })

        # Compute air temperature at the canopy interface
        T_AC[i] = ((T_A_K[i] / R_A[i] + T_S[i] / R_S[i] + T_C[i] / R_x[i])
                   / (1 / R_A[i] + 1 / R_S[i] + 1 / R_x[i]))
        T_AC = np.asarray(np.maximum(1e-3, T_AC))

        # Calculate canopy sensible heat flux (Norman et al 1995)
        H_C[i] = rho[i] * c_p[i] * (T_C[i] - T_AC[i]) / R_x[i]
        # Assume no condensation in the canopy (LE_C<0)
        noC = np.logical_and(i, H_C > Rn_C)
        H_C[noC] = Rn_C[noC]
        flag[noC] = F_ZERO_LE_C
        # Assume no thermal inversion in the canopy
        noI = np.logical_and.reduce(
            (i,
             H_C < calc_H_C_PT(
                 Rn_C,
                 f_g,
                 T_A_K,
                 p,
                 c_p,
                 alpha_PT),
                Rn_C > 0))
        H_C[noI] = 0
        flag[noI] = F_ZERO_H_C

        # Calculate soil sensible heat flux (Norman et al 1995)
        H_S[i] = rho[i] * c_p[i] * (T_S[i] - T_AC[i]) / R_S[i]
        # Assume that there is no condensation in the soil (LE_S<0)
        noC = np.logical_and.reduce((i, H_S > Rn_S - G, (Rn_S - G) > 0))
        H_S[noC] = Rn_S[noC] - G[noC]
        flag[noC] = F_ZERO_LE_S
        # Assume no thermal inversion in the soil
        noI = np.logical_and.reduce((i, H_S < 0, Rn_S - G > 0))
        H_S[noI] = 0
        flag[noI] = F_ZERO_H_S

        # Evaporation Rate (Kustas and Norman 1999)
        H = np.asarray(H_S + H_C)
        LE = np.asarray(Rn - G - H)
        # Now L can be recalculated and the difference between iterations
        # derived
        if const_L is None:
            L[i] = MO.calc_L(u_friction[i],
                             T_A_K[i],
                             rho[i],
                             c_p[i],
                             H[i],
                             LE[i])

            i, l_queue, l_converged, l_diff_max = monin_obukhov_convergence(L,
                                                                            l_queue,
                                                                            l_converged,
                                                                            flag)

    # Compute soil and canopy latent heat fluxes
    LE_S = Rn_S - G - H_S
    LE_C = Rn_C - H_C

    (flag, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L,
     n_iterations) = map(np.asarray, (flag, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S,
                                      G, R_S, R_x, R_A, u_friction, L, iterations))

    return (flag, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L,
            n_iterations)


def TSEB_PT(Tr_K,
            vza,
            T_A_K,
            u,
            ea,
            p,
            Sn_C,
            Sn_S,
            L_dn,
            LAI,
            h_C,
            emis_C,
            emis_S,
            z_0M,
            d_0,
            z_u,
            z_T,
            leaf_width=0.1,
            z0_soil=0.01,
            alpha_PT=1.26,
            x_LAD=1,
            f_c=1.0,
            f_g=1.0,
            w_C=1.0,
            resistance_form=None,
            calcG_params=None,
            const_L=None,
            kB=KB_1_DEFAULT,
            massman_profile=None,
            verbose=True):
    '''Priestley-Taylor TSEB

    Calculates the Priestley Taylor TSEB fluxes using a single observation of
    composite radiometric temperature and using resistances in series.

    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    h_C : float
        Canopy height (m).
    emis_C : float
        Leaf emissivity.
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
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
    w_C : float, optional
        Canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.

            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.

    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with
                                                       G_param list of parameters
                                                       (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.

    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
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
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
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

    # Convert input float scalars to arrays and parameters size
    if calcG_params is None:
        calcG_params = [[1], 0.35]
    if resistance_form is None:
        resistance_form = [0, {}]
    if massman_profile is None:
        massman_profile = [0, []]

    Tr_K = np.asarray(Tr_K, dtype=np.float32)
    (vza,
     T_A_K,
     u,
     ea,
     p,
     Sn_C,
     Sn_S,
     L_dn,
     LAI,
     h_C,
     emis_C,
     emis_S,
     z_0M,
     d_0,
     z_u,
     z_T,
     leaf_width,
     z0_soil,
     alpha_PT,
     x_LAD,
     f_c,
     f_g,
     w_C,
     calcG_array) = map(_check_default_parameter_size,
                        [vza,
                         T_A_K,
                         u,
                         ea,
                         p,
                         Sn_C,
                         Sn_S,
                         L_dn,
                         LAI,
                         h_C,
                         emis_C,
                         emis_S,
                         z_0M,
                         d_0,
                         z_u,
                         z_T,
                         leaf_width,
                         z0_soil,
                         alpha_PT,
                         x_LAD,
                         f_c,
                         f_g,
                         w_C,
                         calcG_params[1]],
                        [Tr_K] * 24)
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    # calcG_params[1] = None
    # Create the output variables
    [Ln_S, Ln_C, H, LE, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, delta_Rn,
     Rn_S, iterations] = [np.zeros(Tr_K.shape, np.float32)+np.nan for i in range(15)]

    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.zeros(Tr_K.shape) + np.inf
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.ones(Tr_K.shape) * const_L
        max_iterations = 1  # No iteration
    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air
    z_0H = res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport

    # Calculate LAI dependent parameters for dataset where LAI > 0
    omega0 = CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c, dtype=np.float32)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta = calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)
    del vza, ea
    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(U_FRICTION_MIN, u_friction), dtype=np.float32)
    L_queue = deque([np.array(L, np.float32)], 6)
    L_converged = np.zeros(Tr_K.shape, bool)
    L_diff_max = np.inf

    # First assume that canopy temperature equals the minumum of Air or
    # radiometric T
    T_C = np.asarray(np.minimum(Tr_K, T_A_K), dtype=np.float32)
    flag, T_S = calc_T_S(Tr_K, T_C, f_theta)
    T_AC = T_A_K.copy()

    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = flag != F_INVALID
        if np.all(L_converged[i]):
            if verbose:
                if L_converged[i].size == 0:
                    print("Finished iterations with no valid solution")
                else:
                    print(f"Finished interations with a max. L diff: {L_diff_max}")
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        if verbose:
            print("Iteration: %d, non-converged pixels: %d, max L diff: %f, total time: %f, loop time: %f" %
                  (n_iterations, np.sum(~L_converged[i]), L_diff_max, total_duration, loop_duration))
        iterations[np.logical_and(~L_converged, flag != F_INVALID)] = n_iterations

        # Inner loop to iterativelly reduce alpha_PT in case latent heat flux
        # from the soil is negative. The initial assumption is of potential
        # canopy transpiration.
        flag[np.logical_and(~L_converged, flag != F_INVALID)] = F_ALL_FLUXES
        LE_S[np.logical_and(~L_converged, flag != F_INVALID)] = -1
        alpha_PT_rec = np.asarray(alpha_PT + 0.1, dtype=np.float32)
        while np.any(LE_S[i] < 0):
            i = np.logical_and.reduce((LE_S < 0, ~L_converged, flag != F_INVALID))

            alpha_PT_rec[i] -= 0.1

            # There cannot be negative transpiration from the vegetation
            alpha_PT_rec[alpha_PT_rec <= 0.0] = 0.0
            flag[np.logical_and(i, alpha_PT_rec == 0.0)] = F_ZERO_LE

            flag[np.logical_and.reduce((i, alpha_PT_rec < alpha_PT, alpha_PT_rec > 0.0))] =\
                F_ZERO_LE_S

            # Calculate aerodynamic resistances
            R_A[i], R_x[i], R_S[i] = calc_resistances(
                      resistance_form,
                      {"R_A": {"z_T": z_T[i], "u_friction": u_friction[i], "L": L[i],
                               "d_0": d_0[i], "z_0H": z_0H[i]},
                       "R_x": {"u_friction": u_friction[i], "h_C": h_C[i],
                               "d_0": d_0[i],
                               "z_0M": z_0M[i], "L": L[i], "F": F[i], "LAI": LAI[i],
                               "leaf_width": leaf_width[i],
                               "z0_soil": z0_soil[i],
                               "massman_profile": massman_profile,
                               "res_params": {k: res_params[k][i] for k in res_params.keys()}},
                       "R_S": {"u_friction": u_friction[i], "h_C": h_C[i],
                               "d_0": d_0[i],
                               "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i],
                               "LAI": LAI[i], "leaf_width": leaf_width[i],
                               "z0_soil": z0_soil[i], "z_u": z_u[i],
                               "deltaT": T_S[i] - T_AC[i], 'u': u[i], 'rho': rho[i],
                               "c_p": c_p[i], "f_cover": f_c[i], "w_C": w_C[i],
                               "massman_profile": massman_profile,
                               "res_params": {k: res_params[k][i] for k in res_params.keys()}}
                       }
            )

            # Calculate net longwave radiation with current values of T_C and T_S
            Ln_C[i], Ln_S[i] = rad.calc_L_n_Campbell(
                T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i], x_LAD=x_LAD[i])
            delta_Rn[i] = Sn_C[i] + Ln_C[i]
            Rn_S[i] = Sn_S[i] + Ln_S[i]

            # Calculate the canopy and soil temperatures using the Priestley
            # Taylor appoach
            H_C[i] = calc_H_C_PT(
                delta_Rn[i],
                f_g[i],
                T_A_K[i],
                p[i],
                c_p[i],
                alpha_PT_rec[i])
            T_C[i] = calc_T_C_series(Tr_K[i], T_A_K[i], R_A[i], R_x[i], R_S[i],
                                     f_theta[i], H_C[i], rho[i], c_p[i])

            # Calculate soil temperature
            flag_t = np.zeros(flag.shape) + F_ALL_FLUXES
            flag_t[i], T_S[i] = calc_T_S(Tr_K[i], T_C[i], f_theta[i])
            flag[flag_t == F_INVALID] = F_INVALID
            LE_S[flag_t == F_INVALID] = 0

            # Recalculate soil resistance using new soil temperature
            _, _, R_S[i] = calc_resistances(
                    resistance_form,
                    {"R_S": {"u_friction": u_friction[i], "h_C": h_C[i], "d_0": d_0[i],
                             "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i],
                             "LAI": LAI[i], "leaf_width": leaf_width[i],
                             "z0_soil": z0_soil[i],  "z_u": z_u[i],
                             "deltaT": T_S[i] - T_AC[i], "u": u[i], "rho": rho[i],
                             "c_p": c_p[i], "f_cover": f_c[i], "w_C": w_C[i],
                             "massman_profile": massman_profile,
                             "res_params": {k: res_params[k][i] for k in res_params.keys()}}
                     }
            )

            i = np.logical_and.reduce((LE_S < 0, ~L_converged, flag != F_INVALID))

            # Get air temperature at canopy interface
            T_AC[i] = ((T_A_K[i] / R_A[i] + T_S[i] / R_S[i] + T_C[i] / R_x[i])
                       / (1.0 / R_A[i] + 1.0 / R_S[i] + 1.0 / R_x[i]))

            # Calculate soil fluxes
            H_S[i] = rho[i] * c_p[i] * (T_S[i] - T_AC[i]) / R_S[i]

            # Compute Soil Heat Flux Ratio
            G[i] = calc_G([calcG_params[0], calcG_array], Rn_S, i)

            # Estimate latent heat fluxes as residual of energy balance at the
            # soil and the canopy
            LE_S[i] = Rn_S[i] - G[i] - H_S[i]
            LE_C[i] = delta_Rn[i] - H_C[i]

            # Special case if there is no transpiration from vegetation.
            # In that case, there should also be no evaporation from the soil
            # and the energy at the soil should be conserved.
            # See end of appendix A1 in Guzinski et al. (2015).
            noT = np.logical_and(i, LE_C == 0)
            H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
            LE_S[noT] = 0

            # Calculate total fluxes
            H[i] = np.asarray(H_C[i] + H_S[i], dtype=np.float32)
            LE[i] = np.asarray(LE_C[i] + LE_S[i], dtype=np.float32)
            # Now L can be recalculated and the difference between iterations
            # derived
            if const_L is None:
                L[i] = MO.calc_L(
                    u_friction[i],
                    T_A_K[i],
                    rho[i],
                    c_p[i],
                    H[i],
                    LE[i])
                # Calculate again the friction velocity with the new stability
                # correctios
                u_friction[i] = MO.calc_u_star(
                    u[i], z_u[i], L[i], d_0[i], z_0M[i])
                u_friction[i] = np.asarray(np.maximum(U_FRICTION_MIN, u_friction[i]), dtype=np.float32)

        if const_L is None:
            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            i, L_queue, L_converged, L_diff_max = monin_obukhov_convergence(L,
                                                                            L_queue,
                                                                            L_converged,
                                                                            flag)

    (flag,
     T_S,
     T_C,
     T_AC,
     L_nS,
     L_nC,
     LE_C,
     H_C,
     LE_S,
     H_S,
     G,
     R_S,
     R_x,
     R_A,
     u_friction,
     L,
     n_iterations) = map(np.asarray,
                         (flag,
                          T_S,
                          T_C,
                          T_AC,
                          Ln_S,
                          Ln_C,
                          LE_C,
                          H_C,
                          LE_S,
                          H_S,
                          G,
                          R_S,
                          R_x,
                          R_A,
                          u_friction,
                          L,
                          iterations))

    return (flag, T_S, T_C, T_AC, L_nS, L_nC, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction,
            L, n_iterations)

def TSEB_SW(Tr_K,
            vza,
            T_A_K,
            u,
            ea,
            p,
            Sn_C,
            Sn_S,
            L_dn,
            LAI,
            h_C,
            emis_C,
            emis_S,
            z_0M,
            d_0,
            z_u,
            z_T,
            leaf_width=0.1,
            z0_soil=0.01,
            Rst_min=100,
            Rss_min=500,
            x_LAD=1,
            f_c=1.0,
            f_g=1.0,
            w_C=1.0,
            resistance_form=None,
            calcG_params=None,
            const_L=None,
            massman_profile=None,
            leaf_type=2,
            kB=KB_1_DEFAULT):
    '''Shuttleworth & Wallace TSEB

    Calculates the Shuttleworth & Wallace TSEB fluxes using a single observation of
    composite radiometric temperature and using resistances in series.

    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    h_C : float
        Canopy height (m).
    emis_C : float
        Leaf emissivity.
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
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
    w_C : float, optional
        Canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.

            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.

    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.

    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
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
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
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
    if massman_profile is None:
        massman_profile = [0, []]
    if calcG_params is None:
        calcG_params = [[1], 0.35]
    if resistance_form is None:
        resistance_form = [0, {}]

    # Convert input float scalars to arrays and parameters size
    Tr_K = np.asarray(Tr_K)
    (vza,
     T_A_K,
     u,
     ea,
     p,
     Sn_C,
     Sn_S,
     L_dn,
     LAI,
     h_C,
     emis_C,
     emis_S,
     z_0M,
     d_0,
     z_u,
     z_T,
     leaf_width,
     z0_soil,
     Rst_min,
     Rss_min,
     x_LAD,
     f_c,
     f_g,
     w_C,
     leaf_type,
     calcG_array) = map(_check_default_parameter_size,
                        [vza,
                         T_A_K,
                         u,
                         ea,
                         p,
                         Sn_C,
                         Sn_S,
                         L_dn,
                         LAI,
                         h_C,
                         emis_C,
                         emis_S,
                         z_0M,
                         d_0,
                         z_u,
                         z_T,
                         leaf_width,
                         z0_soil,
                         Rst_min,
                         Rss_min,
                         x_LAD,
                         f_c,
                         f_g,
                         w_C,
                         leaf_type,
                         calcG_params[1]],
                        [Tr_K] * 26)
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    # calcG_params[1] = None
    # Create the output variables
    [ H, LE, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A,
     Rss_out, Rst_out, iterations, R_c] = [np.zeros(Tr_K.shape)+np.nan for i in range(14)]
    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(Tr_K.shape) + np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(Tr_K.shape) * const_L)
        max_iterations = 1  # No iteration
    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air
    z_0H = res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport
    delta = 10. * met.calc_delta_vapor_pressure(T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_= met.calc_lambda(T_A_K)                     # latent heat of vaporization MJ kg-1
    psicr = met.calc_psicr(c_p, p, lambda_)                     # Psicrometric constant (mb K-1)
    es = met.calc_vapor_pressure(T_A_K)             # saturation water vapour pressure in mb

    rho_cp = rho * c_p
    vpd = es - ea

    # Calculate LAI dependent parameters for dataset where LAI > 0
    omega0 = CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta = calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)
    del vza
    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(U_FRICTION_MIN, u_friction))
    L_queue = deque([np.array(L)], 6)
    L_converged = np.asarray(np.zeros(Tr_K.shape)).astype(bool)
    L_diff_max = np.inf

    # First assume that canopy temperature equals the minumum of Air or
    # radiometric T
    T_C = np.asarray(np.minimum(Tr_K, T_A_K))
    flag, T_S = calc_T_S(Tr_K, T_C, f_theta)
    T_AC = T_A_K.copy()

    _, _, _, taudl = rad.calc_spectra_Cambpell(LAI,
                                               np.zeros(emis_C.shape),
                                               1.0 - emis_C,
                                               np.zeros(emis_S.shape),
                                               1.0 - emis_S,
                                               x_lad=x_LAD,
                                               lai_eff=None)

    emiss = taudl * emis_S + (1 - taudl) * emis_C

    Ln = emiss * (L_dn - met.calc_stephan_boltzmann(T_AC))
    Ln_C = (1. - taudl) * Ln
    Ln_S = taudl * Ln
    delta_Rn = Sn_C + Ln_C
    Rn_S = Sn_S + Ln_S
    Rn = delta_Rn + Rn_S

    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = flag != F_INVALID
        if np.all(L_converged[i]):
            if L_converged[i].size == 0:
                print("Finished iterations with no valid solution")
            else:
                print("Finished interations with a max. L diff: " + str(L_diff_max))
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        print("Iteration: %d, non-converged pixels: %d, max L diff: %f, total time: %f, loop time: %f" %
              (n_iterations, np.sum(~L_converged[i]), L_diff_max, total_duration, loop_duration))
        iterations[np.logical_and(~L_converged, flag != F_INVALID)] = n_iterations

        # Inner loop to iterativelly reduce alpha_PT in case latent heat flux
        # from the soil is negative. The initial assumption is of potential
        # canopy transpiration.
        flag[np.logical_and(~L_converged, flag != F_INVALID)] = F_ALL_FLUXES
        LE_S[np.logical_and(~L_converged, flag != F_INVALID)] = -1

        rst_step = STEP_RST
        Rst = Rst_min[:] - STEP_RST
        Rss = Rss_min[:] - STEP_RSS
        while np.any(LE_S[i] < 0):
            i = np.logical_and.reduce((LE_S < 0,
                                       ~L_converged,
                                       flag != F_INVALID,
                                       Rst <= MAX_RST))
            Rst[i] += rst_step
            Rss[i] += STEP_RSS  # Soil is drier and hence we increase soil surface resistance
            rst_step += RELATIVE_INCREASE * rst_step
            # Ensure that for almost wet soil surface T is also maximum
            # Rst[Rss <= 500] = Rst_min[Rss <= 500]

            # There cannot be negative transpiration from the vegetation
            flag[np.logical_and(i, Rst > MAX_RST)] = F_ZERO_LE

            flag[np.logical_and.reduce((i, Rss > 500, Rst < MAX_RST))] =\
                F_ZERO_LE_S

            # Calculate aerodynamic resistances
            R_A[i], R_x[i], R_S[i] = calc_resistances(resistance_form,
                                                      {"R_A": {"z_T": z_T[i],
                                                               "u_friction": u_friction[i],
                                                               "L": L[i],
                                                              "d_0": d_0[i],
                                                               "z_0H": z_0H[i]},
                                                       "R_x": {"u_friction": u_friction[i],
                                                               "h_C": h_C[i],
                                                               "d_0": d_0[i],
                                                               "z_0M": z_0M[i],
                                                               "L": L[i],
                                                               "F": F[i],
                                                               "LAI": LAI[i],
                                                               "leaf_width": leaf_width[i],
                                                               "res_params": {k: res_params[k][i] for k in res_params.keys()},
                                                               "massman_profile": massman_profile},
                                                       "R_S": {"u_friction": u_friction[i],
                                                               "h_C": h_C[i],
                                                               "d_0": d_0[i],
                                                               "z_0M": z_0M[i],
                                                               "L": L[i],
                                                               "F": F[i],
                                                               "omega0": omega0[i],
                                                               "LAI": LAI[i],
                                                               "leaf_width": leaf_width[i],
                                                               "z0_soil": z0_soil[i],
                                                               "z_u": z_u[i],
                                                               "deltaT": T_S[i] - T_AC[i],
                                                               'u': u[i],
                                                               'rho': rho[i],
                                                               "c_p": c_p[i],
                                                               "f_cover": f_c[i],
                                                               "w_C": w_C[i],
                                                               "res_params": {k: res_params[k][i] for k in res_params.keys()},
                                                               "massman_profile": massman_profile}
                                                       }
                                                      )

            R_c[i] = pet.bulk_stomatal_resistance(LAI[i] * f_g[i], Rst[i], leaf_type=leaf_type[i])
            # Calculate the canopy and soil temperatures using the Priestley Taylor approach
            _, _, _, C_s, C_c = pet.calc_effective_resistances_SW(R_A[i],
                                                                        R_x[i],
                                                                        R_S[i],
                                                                        R_c[i],
                                                                        Rss[i],
                                                                        delta[i],
                                                                        psicr[i])


            # Compute Soil Heat Flux Ratio
            G[i] = calc_G([calcG_params[0], calcG_array], Rn_S, i)

            # Eq. 12 in [Shuttleworth1988]_
            PM_C = (delta[i] * (Rn[i] - G[i]) + (rho_cp[i] * vpd[i] - delta[i] * R_x[i] * (Rn_S[i] - G[i])) / (
                    R_A[i] + R_x[i])) / \
                      (delta[i] + psicr[i] * (1. + R_c[i] / (R_A[i] + R_x[i])))
            PM_C[np.isnan(PM_C)] = 0
            # Eq. 13 in [Shuttleworth1988]_
            PM_S = (delta[i] * (Rn[i] - G[i]) + (rho_cp[i] * vpd[i] - delta[i] * R_S[i] * delta_Rn[i]) / (
                        R_A[i] + R_S[i])) / \
                      (delta[i] + psicr[i] * (1. + Rss[i] / (R_A[i] + R_S[i])))
            PM_S[np.isnan(PM_S)] = 0
            # Eq. 11 in [Shuttleworth1988]_
            LE[i] = C_c * PM_C + C_s * PM_S
            H[i] = Rn[i] - G[i] - LE[i]

            # Compute canopy and soil  fluxes
            # Vapor pressure deficit at canopy source height (mb) # Eq. 8 in [Shuttleworth1988]_
            vpd_0 = vpd[i] + (delta[i] * (Rn[i] - G[i]) - (delta[i] + psicr[i]) * LE[i]) * R_A[i] / (rho_cp[i])
            # Eq. 10 in Shuttleworth & Wallace 1985
            LE_C[i] = (delta[i] * delta_Rn[i] + rho_cp[i] * vpd_0 / R_x[i]) / \
                      (delta[i] + psicr[i] * (1. + R_c[i] / R_x[i]))

            H_C[i] = delta_Rn[i] - LE_C[i]

            T_C[i] = calc_T_C_series(Tr_K[i], T_A_K[i], R_A[i], R_x[i],
                                     R_S[i], f_theta[i], H_C[i], rho[i], c_p[i])

            # Calculate soil temperature
            flag_t = np.zeros(flag.shape) + F_ALL_FLUXES
            flag_t[i], T_S[i] = calc_T_S(Tr_K[i], T_C[i], f_theta[i])
            flag[flag_t == F_INVALID] = F_INVALID
            LE_S[flag_t == F_INVALID] = 0

            # Calculate net longwave radiation with current values of T_C and T_S
            Ln_C[i], Ln_S[i] = rad.calc_L_n_Campbell(
                T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i], x_LAD=x_LAD[i])

            delta_Rn[i] = Sn_C[i] + Ln_C[i]
            Rn_S[i] = Sn_S[i] + Ln_S[i]
            Rn[i] = delta_Rn[i] + Rn_S[i]

            # Recalculate soil resistance using new soil temperature
            _, _, R_S[i] = calc_resistances(resistance_form, {"R_S": {"u_friction": u_friction[i],
                                                                      "h_C": h_C[i],
                                                                      "d_0": d_0[i],
                                                                      "z_0M": z_0M[i],
                                                                      "L": L[i],
                                                                      "F": F[i],
                                                                      "omega0": omega0[i],
                                                                      "LAI": LAI[i],
                                                                      "leaf_width": leaf_width[i],
                                                                      "z0_soil": z0_soil[i],
                                                                      "z_u": z_u[i],
                                                                      "deltaT": T_S[i] - T_AC[i],
                                                                      "u": u[i],
                                                                      "rho": rho[i],
                                                                      "c_p": c_p[i],
                                                                      "f_cover": f_c[i],
                                                                      "w_C": w_C[i],
                                                                      "res_params": {k: res_params[k][i] for k in res_params.keys()},
                                                                      "massman_profile": massman_profile}
                                                              }
                                            )

            i = np.logical_and.reduce((LE_S < 0, ~L_converged, flag != F_INVALID))
            # # Get air temperature at canopy interface
            T_AC[i] = ((T_A_K[i] / R_A[i] + T_S[i] / R_S[i] + T_C[i] / R_x[i])
                       / (1.0 / R_A[i] + 1.0 / R_S[i] + 1.0 / R_x[i]))

            # Calculate heat fluxes
            H_S[i] = rho[i] * c_p[i] * (T_S[i] - T_AC[i]) / R_S[i]
            H_C[i] = rho[i] * c_p[i] * (T_C[i] - T_AC[i]) / R_x[i]

            # Estimate latent heat fluxes as residual of energy balance at the
            # soil and the canopy
            LE_S[i] = Rn_S[i] - G[i] - H_S[i]
            LE_C[i] = delta_Rn[i] - H_C[i]

            # Special case if there is no transpiration from vegetation.
            # In that case, there should also be no evaporation from the soil
            # and the energy at the soil should be conserved.
            # See end of appendix A1 in Guzinski et al. (2015).
            noT = np.logical_and(i, Rst > MAX_RST)
            H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
            LE_S[noT] = 0

            # Calculate total fluxes
            H[i] = np.asarray(H_C[i] + H_S[i])
            LE[i] = np.asarray(LE_C[i] + LE_S[i])

            # Transfer the resistances
            Rst_out[i] = Rst[i]
            Rss_out[i] = Rss[i]
            # Now L can be recalculated and the difference between iterations
            # derived
            if const_L is None:
                L[i] = MO.calc_L(
                    u_friction[i],
                    T_A_K[i],
                    rho[i],
                    c_p[i],
                    H[i],
                    LE[i])
                # Calculate again the friction velocity with the new stability
                # correctios
                u_friction[i] = MO.calc_u_star(
                    u[i], z_u[i], L[i], d_0[i], z_0M[i])
                u_friction[i] = np.asarray(np.maximum(U_FRICTION_MIN, u_friction[i]))

        if const_L is None:
            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            i, L_queue, L_converged, L_diff_max = monin_obukhov_convergence(L,
                                                                         L_queue,
                                                                         L_converged,
                                                                         flag)

    (flag,
     T_S,
     T_C,
     T_AC,
     L_nS,
     L_nC,
     LE_C,
     H_C,
     LE_S,
     H_S,
     G,
     R_S,
     R_x,
     R_A,
     Rss_out,
     Rst_out,
     u_friction,
     L,
     n_iterations) = map(np.asarray,
                         (flag,
                          T_S,
                          T_C,
                          T_AC,
                          Ln_S,
                          Ln_C,
                          LE_C,
                          H_C,
                          LE_S,
                          H_S,
                          G,
                          R_S,
                          R_x,
                          R_A,
                          Rss_out,
                          Rst_out,
                          u_friction,
                          L,
                          iterations))

    return (flag, T_S, T_C, T_AC, L_nS, L_nC, LE_C, H_C, LE_S, H_S, G, R_S, R_x,
            R_A, Rss_out, Rst_out, u_friction, L, n_iterations)


def TSEB_PM(Tr_K,
            vza,
            T_A_K,
            u,
            ea,
            p,
            Sn_C,
            Sn_S,
            L_dn,
            LAI,
            h_C,
            emis_C,
            emis_S,
            z_0M,
            d_0,
            z_u,
            z_T,
            leaf_width=0.1,
            z0_soil=0.01,
            r_c_min=50.,
            x_LAD=1,
            f_c=1.0,
            f_g=1.0,
            w_C=1.0,
            resistance_form=None,
            calcG_params=None,
            const_L=None,
            massman_profile=None,
            kB=KB_1_DEFAULT):
    '''Shuttleworth & Wallace TSEB

    Calculates the Shuttleworth & Wallace TSEB fluxes using a single observation of
    composite radiometric temperature and using resistances in series.

    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    h_C : float
        Canopy height (m).
    emis_C : float
        Leaf emissivity.
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
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
    w_C : float, optional
        Canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.

            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.

    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.

    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
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
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
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


    if massman_profile is None:
        massman_profile = [0, []]
    if calcG_params is None:
        calcG_params = [[1], 0.35]
    if resistance_form is None:
        resistance_form = [0, {}]

    # Convert input float scalars to arrays and parameters size
    Tr_K = np.asarray(Tr_K)
    (vza,
     T_A_K,
     u,
     ea,
     p,
     Sn_C,
     Sn_S,
     L_dn,
     LAI,
     h_C,
     emis_C,
     emis_S,
     z_0M,
     d_0,
     z_u,
     z_T,
     leaf_width,
     z0_soil,
     r_c_min,
     x_LAD,
     f_c,
     f_g,
     w_C,
     calcG_array) = map(_check_default_parameter_size,
                        [vza,
                         T_A_K,
                         u,
                         ea,
                         p,
                         Sn_C,
                         Sn_S,
                         L_dn,
                         LAI,
                         h_C,
                         emis_C,
                         emis_S,
                         z_0M,
                         d_0,
                         z_u,
                         z_T,
                         leaf_width,
                         z0_soil,
                         r_c_min,
                         x_LAD,
                         f_c,
                         f_g,
                         w_C,
                         calcG_params[1]],
                        [Tr_K] * 24)
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    # calcG_params[1] = None
    # Create the output variables
    [flag, H, LE, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A,
     iterations, R_c] = [np.zeros(Tr_K.shape)+np.nan for i in range(13)]

    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(Tr_K.shape) + np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(Tr_K.shape) * const_L)
        max_iterations = 1  # No iteration
    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air
    z_0H = res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport
    delta = 10. * met.calc_delta_vapor_pressure(T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_= met.calc_lambda(T_A_K)                     # latent heat of vaporization MJ kg-1
    psicr = met.calc_psicr(c_p, p, lambda_)                     # Psicrometric constant (mb K-1)
    es = met.calc_vapor_pressure(T_A_K)             # saturation water vapour pressure in mb

    rho_cp = rho * c_p
    vpd = es - ea

    # Calculate LAI dependent parameters for dataset where LAI > 0
    omega0 = CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta = calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)
    del vza
    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(U_FRICTION_MIN, u_friction))
    L_queue = deque([np.array(L)], 6)
    L_converged = np.asarray(np.zeros(Tr_K.shape)).astype(bool)
    L_diff_max = np.inf

    # First assume that canopy temperature equals the minumum of Air or
    # radiometric T
    T_C = np.asarray(np.minimum(Tr_K, T_A_K))
    flag, T_S = calc_T_S(Tr_K, T_C, f_theta)
    T_AC = T_A_K.copy()

    # Calculate net longwave radiation with current values of T_C and T_S
    _, _, _, taudl = rad.calc_spectra_Cambpell(LAI,
                                               np.zeros(emis_C.shape),
                                               1.0 - emis_C,
                                               np.zeros(emis_S.shape),
                                               1.0 - emis_S,
                                               x_lad=x_LAD,
                                               lai_eff=None)
    emiss = taudl * emis_S + (1 - taudl) * emis_C

    Ln = emiss * (L_dn - met.calc_stephan_boltzmann(T_AC))
    Ln_C = (1. - taudl) * Ln
    Ln_S = taudl * Ln
    delta_Rn = Sn_C + Ln_C
    Rn_S = Sn_S + Ln_S
    Rn = delta_Rn + Rn_S

    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = flag != F_INVALID
        if np.all(L_converged[i]):
            if L_converged[i].size == 0:
                print("Finished iterations with no valid solution")
            else:
                print("Finished interations with a max. L diff: " + str(L_diff_max))
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        print("Iteration: %d, non-converged pixels: %d, max L diff: %f, total time: %f, loop time: %f" %
              (n_iterations, np.sum(~L_converged[i]), L_diff_max, total_duration, loop_duration))
        iterations[np.logical_and(~L_converged, flag != F_INVALID)] = n_iterations

        # Inner loop to iterativelly reduce alpha_PT in case latent heat flux
        # from the soil is negative. The initial assumption is of potential
        # canopy transpiration.
        flag[np.logical_and(~L_converged, flag != F_INVALID)] = F_ALL_FLUXES
        LE_S[np.logical_and(~L_converged, flag != F_INVALID)] = -1
        step_rc = STEP_RC
        r_c = np.full(Tr_K.shape, r_c_min - step_rc)

        while np.any(LE_S[i] < 0):
            i = np.logical_and.reduce((LE_S < 0,
                                       ~L_converged,
                                       flag != F_INVALID,
                                       r_c <= MAX_RC))
            r_c[i] += step_rc
            step_rc += RELATIVE_INCREASE * step_rc
            # There cannot be negative transpiration from the vegetation
            flag[np.logical_and(i, r_c > MAX_RC)] = F_ZERO_LE

            # Calculate aerodynamic resistances
            R_A[i], R_x[i], R_S[i] = calc_resistances(resistance_form,
                                                      {"R_A": {"z_T": z_T[i],
                                                               "u_friction": u_friction[i],
                                                               "L": L[i],
                                                              "d_0": d_0[i],
                                                               "z_0H": z_0H[i]},
                                                       "R_x": {"u_friction": u_friction[i],
                                                               "h_C": h_C[i],
                                                               "d_0": d_0[i],
                                                               "z_0M": z_0M[i],
                                                               "L": L[i],
                                                               "F": F[i],
                                                               "LAI": LAI[i],
                                                               "leaf_width": leaf_width[i],
                                                               "res_params": {k: res_params[k][i] for k in res_params.keys()},
                                                               "massman_profile": massman_profile},
                                                       "R_S": {"u_friction": u_friction[i],
                                                               "h_C": h_C[i],
                                                               "d_0": d_0[i],
                                                               "z_0M": z_0M[i],
                                                               "L": L[i],
                                                               "F": F[i],
                                                               "omega0": omega0[i],
                                                               "LAI": LAI[i],
                                                               "leaf_width": leaf_width[i],
                                                               "z0_soil": z0_soil[i],
                                                               "z_u": z_u[i],
                                                               "deltaT": T_S[i] - T_AC[i],
                                                               'u': u[i],
                                                               'rho': rho[i],
                                                               "c_p": c_p[i],
                                                               "f_cover": f_c[i],
                                                               "w_C": w_C[i],
                                                               "res_params": {k: res_params[k][i] for k in res_params.keys()},
                                                               "massman_profile": massman_profile}
                                                       }
                                                      )




            # Compute Soil Heat Flux Ratio
            G[i] = calc_G([calcG_params[0], calcG_array], Rn_S, i)

            # Eq. B1 in [Colaizzi2012]_
            gamma_star = psicr[i] * (1. + r_c[i] / R_A[i])
            LE_C[i] = f_g[i] * (delta[i] * delta_Rn[i] / (delta[i] + gamma_star) +
                        (rho[i] * c_p[i] * vpd[i])
                        / (R_A[i] * (delta[i] + gamma_star)))

            H_C[i] = delta_Rn[i] - LE_C[i]

            T_C[i] = calc_T_C_series(Tr_K[i], T_A_K[i], R_A[i], R_x[i], R_S[i],
                                     f_theta[i], H_C[i], rho[i], c_p[i])


            # Calculate soil temperature
            flag_t = np.zeros(flag.shape) + F_ALL_FLUXES
            flag_t[i], T_S[i] = calc_T_S(Tr_K[i], T_C[i], f_theta[i])
            flag[flag_t == F_INVALID] = F_INVALID
            LE_S[flag_t == F_INVALID] = 0

            # Recalculate soil resistance using new soil temperature
            _, _, R_S[i] = calc_resistances(resistance_form, {"R_S": {"u_friction": u_friction[i],
                                                                      "h_C": h_C[i],
                                                                      "d_0": d_0[i],
                                                                      "z_0M": z_0M[i],
                                                                      "L": L[i],
                                                                      "F": F[i],
                                                                      "omega0": omega0[i],
                                                                      "LAI": LAI[i],
                                                                      "leaf_width": leaf_width[i],
                                                                      "z0_soil": z0_soil[i],
                                                                      "z_u": z_u[i],
                                                                      "deltaT": T_S[i] - T_AC[i],
                                                                      "u": u[i],
                                                                      "rho": rho[i],
                                                                      "c_p": c_p[i],
                                                                      "f_cover": f_c[i],
                                                                      "w_C": w_C[i],
                                                                      "res_params": {k: res_params[k][i] for k in res_params.keys()},
                                                                      "massman_profile": massman_profile}
                                                              }
                                            )

            i = np.logical_and.reduce((LE_S < 0, ~L_converged, flag != F_INVALID))
            # Get air temperature at canopy interface
            T_AC[i] = ((T_A_K[i] / R_A[i] + T_S[i] / R_S[i] + T_C[i] / R_x[i])
                       / (1.0 / R_A[i] + 1.0 / R_S[i] + 1.0 / R_x[i]))

            # Calculate heat fluxes
            H_S[i] = rho[i] * c_p[i] * (T_S[i] - T_AC[i]) / R_S[i]
            H_C[i] = rho[i] * c_p[i] * (T_C[i] - T_AC[i]) / R_x[i]

            # Calculate net longwave radiation with current values of T_C and T_S
            Ln_C[i], Ln_S[i] = rad.calc_L_n_Campbell(
                T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i], x_LAD=x_LAD[i])

            delta_Rn[i] = Sn_C[i] + Ln_C[i]
            Rn_S[i] = Sn_S[i] + Ln_S[i]
            Rn[i] = delta_Rn[i] + Rn_S[i]
            # Estimate latent heat fluxes as residual of energy balance at the
            # soil and the canopy
            LE_S[i] = Rn_S[i] - G[i] - H_S[i]
            LE_C[i] = delta_Rn[i] - H_C[i]

            # Special case if there is no transpiration from vegetation.
            # In that case, there should also be no evaporation from the soil
            # and the energy at the soil should be conserved.
            # See end of appendix A1 in Guzinski et al. (2015).
            noT = np.logical_and(i, r_c > MAX_RC)
            H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
            LE_S[noT] = 0

            # Calculate total fluxes
            H[i] = np.asarray(H_C[i] + H_S[i])
            LE[i] = np.asarray(LE_C[i] + LE_S[i])
            # Now L can be recalculated and the difference between iterations
            # derived
            if const_L is None:
                L[i] = MO.calc_L(u_friction[i], T_A_K[i], rho[i], c_p[i], H[i], LE[i])
                # Calculate again the friction velocity with the new stability
                # corrections
                u_friction[i] = MO.calc_u_star(
                    u[i], z_u[i], L[i], d_0[i], z_0M[i])
                u_friction[i] = np.asarray(np.maximum(U_FRICTION_MIN, u_friction[i]))

        if const_L is None:
            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            i, L_queue, L_converged, L_diff_max = monin_obukhov_convergence(L,
                                                                         L_queue,
                                                                         L_converged,
                                                                         flag)

    (flag,
     T_S,
     T_C,
     T_AC,
     L_nS,
     L_nC,
     LE_C,
     H_C,
     LE_S,
     H_S,
     G,
     R_S,
     R_x,
     R_A,
     u_friction,
     L,
     n_iterations) = map(np.asarray,
                         (flag,
                          T_S,
                          T_C,
                          T_AC,
                          Ln_S,
                          Ln_C,
                          LE_C,
                          H_C,
                          LE_S,
                          H_S,
                          G,
                          R_S,
                          R_x,
                          R_A,
                          u_friction,
                          L,
                          iterations))

    return (flag, T_S, T_C, T_AC, L_nS, L_nC, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction,
            L, n_iterations)


def _L_diff(L, L_old):
    L_diff = np.asarray(np.fabs(L - L_old) / np.fabs(L_old), dtype=np.float32)
    L_diff[np.isnan(L_diff)] = float('inf')
    return L_diff


def DTD(Tr_K_0,
        Tr_K_1,
        vza,
        T_A_K_0,
        T_A_K_1,
        u,
        ea,
        p,
        Sn_C,
        Sn_S,
        L_dn,
        LAI,
        h_C,
        emis_C,
        emis_S,
        z_0M,
        d_0,
        z_u,
        z_T,
        leaf_width=0.1,
        z0_soil=0.01,
        alpha_PT=1.26,
        x_LAD=1,
        f_c=1.0,
        f_g=1.0,
        w_C=1.0,
        resistance_form=None,
        calcG_params=None,
        calc_Ri=True,
        kB=KB_1_DEFAULT,
        massman_profile=None,
        verbose=True):
    ''' Calculate daytime Dual Time Difference TSEB fluxes

    Parameters
    ----------
    Tr_K_0 : float
        Radiometric composite temperature around sunrise(Kelvin).
    Tr_K_1 : float
        Radiometric composite temperature near noon (Kelvin).
    vza : float
        View Zenith Angle near noon (degrees).
    T_A_K_0 : float
        Air temperature around sunrise (Kelvin).
    T_A_K_1 : float
        Air temperature near noon (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    h_C : float
        Canopy height (m).
    emis_C : float
        Leaf emissivity.
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
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
    w_C : Optional[float]
        Canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.

            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.

    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with
                                                       G_param list of parameters
                                                       (see :func:`~TSEB.calc_G_time_diff`).
    calc_Ri : float or None, optional
        If included, its value will be used to force the Richardson Number.

    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
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
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
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

    # Convert input scalars to numpy arrays and parameters size
    if calcG_params is None:
        calcG_params = [[1], 0.35]
    if resistance_form is None:
        resistance_form = [0, {}]
    if massman_profile is None:
        massman_profile = [0, []]

    Tr_K_0 = np.asarray(Tr_K_0)
    (Tr_K_1,
     vza,
     T_A_K_0,
     T_A_K_1,
     u,
     ea,
     p,
     Sn_C,
     Sn_S,
     L_dn,
     LAI,
     h_C,
     emis_C,
     emis_S,
     z_0M,
     d_0,
     z_u,
     z_T,
     leaf_width,
     z0_soil,
     alpha_PT,
     x_LAD,
     f_c,
     f_g,
     w_C,
     calcG_array) = map(_check_default_parameter_size,
                        [Tr_K_1,
                         vza,
                         T_A_K_0,
                         T_A_K_1,
                         u,
                         ea,
                         p,
                         Sn_C,
                         Sn_S,
                         L_dn,
                         LAI,
                         h_C,
                         emis_C,
                         emis_S,
                         z_0M,
                         d_0,
                         z_u,
                         z_T,
                         leaf_width,
                         z0_soil,
                         alpha_PT,
                         x_LAD,
                         f_c,
                         f_g,
                         w_C,
                         calcG_params[1]],
                        [Tr_K_0] * 26)
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    # Create the output variables
    [flag, T_S, T_C, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x,
        R_A, H, iterations] = [np.zeros(Tr_K_1.shape, np.float32) + np.nan for i in range(16)]

    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K_1)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air
    z_0H = res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport

    # Calculate LAI dependent parameters for dataset where LAI > 0
    # Clumping factor at nadir
    omega0 = CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta = calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)

    # L is not used in the DTD, since Richardson number is used instead to
    # avoid dependance on non-differential temperatures. But it is still saved
    # in the output for testing purposes.
    if isinstance(calc_Ri, bool):
        # Calculate the Richardson number
        Ri = MO.calc_richardson(u, z_u, d_0, Tr_K_0, Tr_K_1, T_A_K_0, T_A_K_1)
    else:  # We force Monin-Obukhov lenght to the provided array/value
        Ri = np.asarray(np.ones(Tr_K_1.shape) * calc_Ri)
    # Use the approximation Ri ~ (z-d_0)./L from end of section 2.2 from
    # Norman et. al., 2000 (DTD paper)
    L_from_Ri = (z_u - d_0) / Ri

    # calculate the resistances
    # First calcualte u_S, wind speed at the soil surface
    u_friction = MO.calc_u_star(u, z_u, L_from_Ri, d_0, z_0M)
    u_friction = np.asarray(np.maximum(U_FRICTION_MIN, u_friction))

    # First assume that canopy temperature equals the minumum of Air or
    # radiometric T
    T_C = np.asarray(np.minimum(Tr_K_1, T_A_K_1))
    flag, T_S = calc_T_S(Tr_K_1, T_C, f_theta)

    # Calculate aerodynamic resistances
    R_A_params = {"z_T": z_T, "u_friction": u_friction,
                  "L": L_from_Ri, "d_0": d_0, "z_0H": z_0H}
    params = {k: res_params[k] for k in res_params.keys()}
    R_x_params = {"u_friction": u_friction, "h_C": h_C, "d_0": d_0,
                  "z_0M": z_0M, "L": L_from_Ri, "F": F, "LAI": LAI,
                  "leaf_width": leaf_width,
                  "z0_soil": z0_soil,
                  "massman_profile": massman_profile,
                  "res_params": params}
    # based on equation from Guzinski et. al., 2015
    deltaT = (Tr_K_1 - Tr_K_0) - (T_A_K_1 - T_A_K_0)
    R_S_params = {"u_friction": u_friction, "h_C": h_C, "d_0": d_0,
                  "z_0M": z_0M, "L": L_from_Ri, "F": F,
                  "omega0": omega0, "LAI": LAI,
                  "leaf_width": leaf_width, "z0_soil": z0_soil, "z_u": z_u,
                  "deltaT": deltaT,
                  "massman_profile": massman_profile,
                  "res_params": params}
    res_types = {"R_A": R_A_params, "R_x": R_x_params, "R_S": R_S_params}
    del R_A_params, R_x_params, R_S_params
    R_A, R_x, R_S = calc_resistances(resistance_form, res_types)
    del res_types

    # Outer loop until canopy and soil temperatures have stabilised
    T_C_prev = np.zeros(Tr_K_1.shape)
    T_C_thres = 0.1
    T_C_diff = np.fabs(T_C - T_C_prev)
    for n_iterations in range(ITERATIONS):
        i = flag != F_INVALID
        if np.all(T_C_diff[i] < T_C_thres):
            if verbose:
                if T_C_diff[i].size == 0:
                    print("Finished iterations with no valid solution")
                else:
                    print(f"Finished iteration with a max. T_C diff: {np.max(T_C_diff[i])}")
            break
        if verbose:
            print(f"Iteration {n_iterations},"
              f"maximum T_C difference between iterations: {np.max(T_C_diff[i])}")
        iterations[np.logical_and(T_C_diff >= T_C_thres, flag != F_INVALID)] = n_iterations

        # Inner loop to iterativelly reduce alpha_PT in case latent heat flux
        # from the soil is negative. The initial assumption is of potential
        # canopy transpiration.
        flag[np.logical_and(T_C_diff >= T_C_thres, flag != F_INVALID)] = F_ALL_FLUXES
        LE_S[np.logical_and(T_C_diff >= T_C_thres, flag != F_INVALID)] = -1
        alpha_PT_rec = np.asarray(alpha_PT + 0.1)

        while np.any(LE_S[i] < 0):
            i = np.logical_and.reduce(
                (LE_S < 0, T_C_diff >= T_C_thres, flag != F_INVALID))

            alpha_PT_rec[i] -= 0.1

            # There cannot be negative transpiration from the vegetation
            alpha_PT_rec[alpha_PT_rec <= 0.0] = 0.0
            flag[np.logical_and(i, alpha_PT_rec == 0.0)] = F_ZERO_LE

            flag[np.logical_and.reduce((i, alpha_PT_rec < alpha_PT, alpha_PT_rec > 0.0))] =\
                F_ZERO_LE_S

            # Calculate net longwave radiation with current values of T_C and T_S
            Ln_C[i], Ln_S[i] = rad.calc_L_n_Campbell(
                T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i], x_LAD=x_LAD[i])

            # Calculate total net radiation of soil and canopy
            delta_Rn = Sn_C + Ln_C
            Rn_S = Sn_S + Ln_S

            # Calculate sensible heat fluxes at time t1
            H_C[i] = calc_H_C_PT(
                delta_Rn[i],
                f_g[i],
                T_A_K_1[i],
                p[i],
                c_p[i],
                alpha_PT_rec[i])
            H[i] = calc_H_DTD_series(
                Tr_K_1[i],
                Tr_K_0[i],
                T_A_K_1[i],
                T_A_K_0[i],
                rho[i],
                c_p[i],
                f_theta[i],
                R_S[i],
                R_A[i],
                R_x[i],
                H_C[i])
            H_S[i] = H[i] - H_C[i]

            # Calculate ground heat flux
            G[i] = calc_G([calcG_params[0], calcG_array], Rn_S, i)

            # Calculate latent heat fluxes as residuals
            LE_S[i] = Rn_S[i] - H_S[i] - G[i]
            LE_C[i] = delta_Rn[i] - H_C[i]

            # Special case if there is no transpiration from vegetation.
            # In that case, there should also be no evaporation from the soil
            # and the energy at the soil should be conserved.
            # See end of appendix A1 in Guzinski et al. (2015).
            noT = np.logical_and(i, LE_C == 0)
            H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
            LE_S[noT] = 0

            # Recalculate soil and canopy temperatures. They are used only for
            # estimation of longwave radiation, so the use of non-differential Tr
            # and T_A shouldn't affect the turbulent fluxes much
            T_C[i] = calc_T_C_series(
                Tr_K_1[i],
                T_A_K_1[i],
                R_A[i],
                R_x[i],
                R_S[i],
                f_theta[i],
                H_C[i],
                rho[i],
                c_p[i])
            flag_t = np.zeros(flag.shape) + F_ALL_FLUXES
            flag_t[i], T_S[i] = calc_T_S(Tr_K_1[i], T_C[i], f_theta[i])
            flag[flag_t == F_INVALID] = F_INVALID
            LE_S[flag_t == F_INVALID] = 0

            # Recalculate soil resistance using new difference between soil
            # and canopy temperatures. deltaT is equivalent to T_S - T_C while
            # not being dependent on non-differential T_A.
            params = {k: res_params[k][i] for k in res_params.keys()}
            deltaT = (H_S[i] * R_S[i] - H_C[i] * R_x[i]) / (rho[i] * c_p[i])
            R_S_params = {"u_friction": u_friction[i], "h_C": h_C[i], "d_0": d_0[i],
                          "z_0M": z_0M[i], "L": L_from_Ri[i], "F": F[i], "omega0": omega0[i],
                          "LAI": LAI[i], "leaf_width": leaf_width[i],
                          "z0_soil": z0_soil[i], "z_u": z_u[i],
                          "deltaT": deltaT,
                          "massman_profile": massman_profile,
                          "res_params": params}
            _, _, R_S[i] = calc_resistances(resistance_form, {"R_S": R_S_params})

        T_C_diff = np.asarray(np.fabs(T_C - T_C_prev))
        T_C_prev = np.array(T_C)

    # L and T_AC are only calculated for testing purposes
    L = MO.calc_L(u_friction, T_A_K_1, rho, c_p, H, LE_C + LE_S)
    T_AC = ((T_A_K_1 / R_A + T_S / R_S + T_C / R_x)
            / (1.0 / R_A + 1.0 / R_S + 1.0 / R_x))

    (flag,
     T_S,
     T_C,
     T_AC,
     L_nS,
     L_nC,
     LE_C,
     H_C,
     LE_S,
     H_S,
     G,
     R_S,
     R_x,
     R_A,
     u_friction,
     L,
     Ri,
     n_iterations) = map(np.asarray,
                         (flag,
                          T_S,
                          T_C,
                          T_AC,
                          Ln_S,
                          Ln_C,
                          LE_C,
                          H_C,
                          LE_S,
                          H_S,
                          G,
                          R_S,
                          R_x,
                          R_A,
                          u_friction,
                          L,
                          Ri,
                          iterations))
    return [
        flag,
        T_S,
        T_C,
        T_AC,
        L_nS,
        L_nC,
        LE_C,
        H_C,
        LE_S,
        H_S,
        G,
        R_S,
        R_x,
        R_A,
        u_friction,
        L,
        Ri,
        n_iterations]


def OSEB(Tr_K,
         T_A_K,
         u,
         ea,
         p,
         Sn,
         L_dn,
         emis,
         z_0M,
         d_0,
         z_u,
         z_T,
         calcG_params=[
             [1],
             0.35],
         const_L=None,
         T0_K=[],
         kB=KB_1_DEFAULT):
    '''Calulates bulk fluxes from a One Source Energy Balance model

    Parameters
    ----------
    Tr_K : float or array
        Radiometric composite temperature (Kelvin).
    T_A_K : float or array
        Air temperature (Kelvin).
    u : float or array
        Wind speed above the canopy (m s-1).
    ea : float or array
        Water vapour pressure above the canopy (mb).
    p : float or array
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn : float or array
        Net shortwave radiation (W m-2).
    L_dn : float or array
        Downwelling longwave radiation (W m-2)
    emis : float or array
        Surface emissivity.
    z_0M : float or array
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float or array
        Zero-plane displacement height (m).
    z_u : float or array
        Height of measurement of windspeed (m).
    z_T : float or array
        Height of measurement of air temperature (m).
    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with
                                                       G_param list of parameters
                                                       (see :func:`~TSEB.calc_G_time_diff`).
    const_L : Optional[float]
        If included, its value will be used to force the Moning-Obukhov stability length.
    T0_K: Optional[tuple(float or array,float or array)]
        If given it contains radiometric composite temperature (K) at time 0 as
        the first element and air temperature (K) at time 0 as the second element,
        in order to derive differential temperatures like is done in DTD


    Returns
    -------
    flag : int or array
        Quality flag, see Appendix for description.
    Ln : float or array
        Net longwave radiation (W m-2)
    LE : float or array
        Latent heat flux (W m-2).
    H : float or array
        Sensible heat flux (W m-2).
    G : float or array
        Soil heat flux (W m-2).
    R_A : float or array
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float or array
        Friction velocity (m s-1).
    L : float or array
        Monin-Obuhkov length (m).
    n_iterations : int or array
        number of iterations until convergence of L.
    '''

    # Convert input scalars to numpy arrays and check parameters size
    Tr_K = np.asarray(Tr_K)
    (T_A_K,
     u,
     ea,
     p,
     Sn,
     L_dn,
     emis,
     z_0M,
     d_0,
     z_u,
     z_T,
     calcG_array) = map(_check_default_parameter_size,
                        [T_A_K,
                         u,
                         ea,
                         p,
                         Sn,
                         L_dn,
                         emis,
                         z_0M,
                         d_0,
                         z_u,
                         z_T,
                         calcG_params[1]],
                        [Tr_K] * 12)
    # Create the output variables
    [flag, Ln, LE, H, G, R_A] = [np.zeros(Tr_K.shape, np.float32) + np.nan for i in range(6)]

    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.zeros(Tr_K.shape) + np.inf
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.ones(Tr_K.shape) * const_L
        max_iterations = 1  # No iteration

    # Check if differential temperatures are to be used
    if len(T0_K) == 2:
        differentialT = True
        Tr_K_0 = np.asarray(T0_K[0])
        T_A_K_0 = np.asarray(T0_K[1])
    else:
        differentialT = False

    # Initially assume stable atmospheric conditions and set variables for
    L_old = np.ones(Tr_K.shape)
    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air

    # With differential temperatures use Richardson number to approximate L,
    # same as is done in DTD
    if differentialT:
        if const_L is None:
            Ri = MO.calc_richardson(u, z_u, d_0, Tr_K_0, Tr_K, T_A_K_0, T_A_K)
        else:
            Ri = np.array(L)
        # Use the approximation Ri ~ (z-d_0)./L from end of section 2.2 from
        # Norman et. al., 2000 (DTD paper)
        L_from_Ri = (z_u - d_0) / Ri
        u_friction = MO.calc_u_star(u, z_u, L_from_Ri, d_0, z_0M)
    else:
        u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.maximum(U_FRICTION_MIN, u_friction)
    L_old = np.ones(Tr_K.shape)
    L_diff = np.ones(Tr_K.shape) * float('inf')

    z_0H = res.calc_z_0H(z_0M, kB=kB)

    # Calculate Net radiation
    Ln = emis * L_dn - emis * met.calc_stephan_boltzmann(Tr_K)
    Rn = np.asarray(Sn + Ln)

    # Compute Soil Heat Flux
    i = np.ones(Rn.shape, dtype=bool)
    G[i] = calc_G([calcG_params[0], calcG_array], Rn, i)

    # Loop for estimating atmospheric stability.
    # Stops when difference in consecutive L and u_friction is below a
    # given threshold
    for n_iterations in range(max_iterations):
        flag = np.zeros(Tr_K.shape) + F_ALL_FLUXES_OS
        # Stop the iteration if differences are below the threshold
        if np.all(L_diff < L_thres):
            break

        # Calculate aerodynamic resistances
        if differentialT:
            R_A_params = {"z_T": z_T, "u_friction": u_friction,
                          "L": L_from_Ri, "d_0": d_0, "z_0H": z_0H}
        else:
            R_A_params = {"z_T": z_T, "u_friction": u_friction,
                          "L": L, "d_0": d_0, "z_0H": z_0H}
        R_A, _, _ = calc_resistances(KUSTAS_NORMAN_1999, {"R_A": R_A_params})

        # Calculate bulk fluxes assuming that since there is no vegetation,
        # Tr is the heat source
        if differentialT:
            H = rho * c_p * ((Tr_K - Tr_K_0) - (T_A_K - T_A_K_0)) / R_A
        else:
            H = rho * c_p * (Tr_K - T_A_K) / R_A
        H = np.asarray(H)
        LE = np.asarray(Rn - G - H)

        # Avoid negative ET during daytime and make sure that energy is
        # conserved
        flag[LE < 0] = F_ZERO_LE_OS
        H[LE < 0] = np.minimum(H[LE < 0], Rn[LE < 0] - G[LE < 0])
        G[LE < 0] = np.maximum(G[LE < 0], Rn[LE < 0] - H[LE < 0])
        LE[LE < 0] = 0

        if const_L is None:
            # Now L can be recalculated and the difference between iterations
            # derived
            L = MO.calc_L(u_friction, T_A_K, rho, c_p, H, LE)
            L_diff = np.fabs(L - L_old) / np.fabs(L_old)
            L_old = np.array(L)
            L_old[np.fabs(L_old) == 0] = 1e-36

            # Calculate again the friction velocity with the new stability correction
            # and derive the change between iterations
            if not differentialT:
                u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
                u_friction = np.maximum(U_FRICTION_MIN, u_friction)

    flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations = map(
        np.asarray, (flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations))

    return flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations


def calc_F_theta_campbell(theta, F, w_C=1, Omega0=1, x_LAD=1):
    '''Calculates the fraction of vegetatinon observed at an angle.

    Parameters
    ----------
    theta : float
        Angle of incidence (degrees).
    F : float
        Real Leaf (Plant) Area Index.
    w_C : float
        Canopy width to height ratio, optional (default = 1).
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
    # Convert from canopy width/height to height/width as required by Kustas' Omega function
    w_C = 1. / w_C
    # First calcualte the angular clumping factor Omega based on eq (3) from
    # W.P. Kustas, J.M. Norman,  Agricultural and Forest Meteorology 94 (1999)
    # CHECK: should theta here be in degrees or radians
    OmegaTheta = (Omega0 / (Omega0 + (1.0 - Omega0)
                  * np.exp(-2.2 * np.radians(theta)**(3.8 - 0.46 * w_C))))
    # Estimate the beam extinction coefficient based on a elipsoidal LAD function
    # Eq. 15.4 of Campbell and Norman (1998)
    K_be = rad.calc_K_be_Campbell(theta, x_LAD)
    ftheta = 1.0 - np.exp(-K_be * OmegaTheta * F)
    return np.asarray(ftheta, dtype=np.float32)


def calc_G(calcG_params, Rn_S, i=None):

    if i is None:
        i = np.ones(Rn_S.shape, dtype=bool)
    if calcG_params[0][0] == G_CONSTANT:
        G = calcG_params[1][i]
    elif calcG_params[0][0] == G_RATIO:
        G = calc_G_ratio(Rn_S[i], calcG_params[1][i])
    elif calcG_params[0][0] == G_TIME_DIFF:
        G = calc_G_time_diff(Rn_S[i],
                             [calcG_params[1][i], calcG_params[0][1],
                              calcG_params[0][2], calcG_params[0][3]])
    elif calcG_params[0][0] == G_TIME_DIFF_SIGMOID:
        G = calc_G_time_diff_sigmoid(Rn_S[i], [calcG_params[1][i], calcG_params[0][1],
                                     calcG_params[0][2], calcG_params[0][3], calcG_params[0][4],
                                     calcG_params[0][5], calcG_params[0][6]])

    return np.asarray(G)


def calc_G_time_diff(R_n, G_param=[12.0, 0.35, 3.0, 24.0]):
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
                shift of peak G relative to solar noon (default 3hrs before noon).
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

    # Get parameters
    time = G_param[0] - 12.0
    A = G_param[1]
    phase_shift = G_param[2]
    B = G_param[3]
    G_ratio = A * np.cos(2.0 * np.pi * (time + phase_shift) / B)
    G = R_n * G_ratio
    return np.asarray(G, dtype=np.float32)


def calc_G_time_diff_sigmoid(R_n, G_param=[12, 0, 0.35, 10.0, 14.0, 1.0, 1.0]):
    ''' Estimates Soil Heat Flux as function of time and net radiation using an asymmetric sigmoid
    function

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

    # Get parameters
    time, G_ratio_min, G_ratio_max, phase_shift_0, phase_shift_1, shape_0, shape_1 = G_param
    G_ratio = (G_ratio_min + (G_ratio_max - G_ratio_min)
               * 0.5 * (np.tanh((time - phase_shift_0) / shape_0)
               - np.tanh((time - phase_shift_1) / shape_1)))
    G = R_n * G_ratio
    return np.asarray(G, dtype=np.float32)


def calc_G_ratio(Rn_S, G_ratio=0.35):
    '''Estimates Soil Heat Flux as ratio of net soil radiation.

    Parameters
    ----------
    Rn_S : float
        Net soil radiation (W m-2).
    G_ratio : float, optional
        G/Rn_S ratio, default=0.35.

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

    G = G_ratio * Rn_S
    return np.asarray(G, dtype=np.float32)


def calc_H_C(T_C, T_A, R_A, rho, c_p):
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

    H_C = rho * c_p * (T_C - T_A) / R_A
    return np.asarray(H_C, dtype=np.float32)


def calc_H_C_PT(delta_R_ni, f_g, T_A_K, P, c_p, alpha):
    '''Calculates canopy sensible heat flux based on the Priestley and Taylor formula.

    Parameters
    ----------
    delta_R_ni : float
        net radiation divergence of the vegetative canopy (W m-2).
    f_g : float
        fraction of vegetative canopy that is green.
    T_A_K : float
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
    s = met.calc_delta_vapor_pressure(T_A_K)
    s = s * 10  # to mb
    # latent heat of vaporisation (J./kg)
    Lambda = met.calc_lambda(T_A_K)
    # psychrometric constant (mb C-1)
    gama = met.calc_psicr(c_p, P, Lambda)
    s_gama = s / (s + gama)
    H_C = delta_R_ni * (1.0 - alpha * f_g * s_gama)
    return np.asarray(H_C, dtype=np.float32)


def calc_H_DTD_parallel(
        T_R1,
        T_R0,
        T_A1,
        T_A0,
        rho,
        c_p,
        f_theta1,
        R_S1,
        R_A1,
        R_AC1,
        H_C1):
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
    .. [Guzinski2013] Guzinski, R., Anderson, M. C., Kustas, W. P., Nieto, H., and Sandholt, I.
        (2013) Using a thermal-based two source energy balance model with time-differencing to
        estimate surface energy fluxes with day-night MODIS observations,
        Hydrol. Earth Syst. Sci., 17, 2809-2825,
        http://dx.doi.org/10.5194/hess-17-2809-2013.
    '''

    # Ignore night fluxes
    H = (rho * c_p * (((T_R1 - T_R0) - (T_A1 - T_A0)) / ((1.0 - f_theta1) * (R_A1 + R_S1)))
         + H_C1 * (1.0 - ((f_theta1 * R_AC1) / ((1.0 - f_theta1) * (R_A1 + R_S1)))))
    return np.asarray(H, dtype=np.float32)


def calc_H_DTD_series(
        T_R1,
        T_R0,
        T_A1,
        T_A0,
        rho,
        c_p,
        f_theta,
        R_S,
        R_A,
        R_x,
        H_C):
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

    H = (rho * c_p * ((T_R1 - T_R0) - (T_A1 - T_A0)) / ((1.0 - f_theta) * R_S + R_A)
         + H_C * ((1.0 - f_theta) * R_S - f_theta * R_x) / ((1.0 - f_theta) * R_S + R_A))
    return np.asarray(H, dtype=np.float32)


def calc_H_S(T_S, T_A, R_A, R_S, rho, c_p):
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

    H_S = rho * c_p * ((T_S - T_A) / (R_S + R_A))
    return np.asarray(H_S, dtype=np.float32)


def calc_T_C(T_R, T_S, f_theta):
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

    # Convert input scalars to numpy array
    (T_R, T_S, f_theta) = map(np.asarray, (T_R, T_S, f_theta))
    T_temp = np.asarray(T_R ** 4 - (1.0 - f_theta) * T_S**4)
    T_C = np.zeros(T_R.shape)
    flag = np.zeros(T_R.shape) + F_ALL_FLUXES

    # Succesfull inversion
    T_C[T_temp >= 0] = (T_temp[T_temp >= 0] / f_theta[T_temp >= 0])**0.25

    # Unsuccesfull inversion
    T_C[T_temp < 0] = 1e-6
    flag[T_temp < 0] = F_INVALID

    return np.asarray(flag), np.asarray(T_C, dtype=np.float32)


def calc_T_C_series(Tr_K, T_A_K, R_A, R_x, R_S, f_theta, H_C, rho, c_p):
    '''Estimates canopy temperature from canopy sensible heat flux and
    resistance network in series.

    Parameters
    ----------
    Tr_K : float
        Directional Radiometric Temperature (K).
    T_A_K : float
        Air Temperature (K).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk aerodynamic resistance to heat transport at the canopy boundary layer (s m-1).
    R_S : float
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
    T_C : float
        Canopy temperature (K).

    References
    ----------
    Eqs. A5-A13 in [Norman1995]_'''

    T_R_K_4 = Tr_K**4
    # equation A7 from Norman 1995, linear approximation of temperature of the
    # canopy
    T_C_lin = ((T_A_K / R_A + Tr_K / (R_S * (1.0 - f_theta))
                + H_C * R_x / (rho * c_p) * (1.0 / R_A + 1.0 / R_S + 1.0 / R_x))
               / (1.0 / R_A + 1.0 / R_S + f_theta / (R_S * (1.0 - f_theta))))
    # equation A12 from Norman 1995
    T_D = (T_C_lin * (1 + R_S / R_A) - H_C * R_x / (rho * c_p)
           * (1.0 + R_S / R_x + R_S / R_A) - T_A_K * R_S / R_A)
    # equation A11 from Norman 1995
    delta_T_C = (((T_R_K_4 - f_theta * T_C_lin**4 - (1.0 - f_theta) * T_D**4)
                 / (4.0 * (1.0 - f_theta) * T_D**3 * (1.0 + R_S / R_A)
                 + 4.0 * f_theta * T_C_lin**3)))
    # get canopy temperature in Kelvin
    T_C = T_C_lin + delta_T_C
    return np.asarray(T_C, dtype=np.float32)


def calc_T_CS_Norman(F, vza_n, vza_f, T_n, T_f, w_C=1, x_LAD=1, omega0=1):
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
    w_C : float,optional
        Canopy height to width ratio, use w_C=1 by default.
    x_LAD : float,optional
        Chi parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    omega0 : float,optional
        Clumping index at nadir, use omega0=1 by default.

    Returns
    -------
    T_C : float
        Canopy temperature (K).
    T_S : float
        Soil temperature (K).

    References
    ----------
    inversion of Eq. 1 in [Norman1995]_
    '''

    # Convert  the input scalars to numpy arrays
    F, vza_n, vza_f, T_n, T_f, w_C, x_LAD, omega0 = map(
        np.asarray, (F, vza_n, vza_f, T_n, T_f, w_C, x_LAD, omega0))
    # Calculate the fraction of vegetation observed by each angle
    f_theta_n = calc_F_theta_campbell(vza_n, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)
    f_theta_f = calc_F_theta_campbell(vza_f, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)
    # Solve the sytem of two unknowns and two equations
    T_S_4 = np.asarray((f_theta_f * T_n**4 - f_theta_n * T_f**4)
                       / (f_theta_f - f_theta_n))
    T_C_4 = np.asarray((T_n ** 4 - (1.0 - f_theta_n) * T_S_4) / f_theta_n)

    T_C_K = np.zeros(T_n.shape)
    T_S_K = np.zeros(T_n.shape)

    # Successful inversion
    i = np.logical_and(T_C_4 > 0, T_S_4 > 0)
    T_C_K[i] = T_C_4[i]**0.25
    T_S_K[i] = T_S_4[i]**0.25

    # Unsuccessful inversion
    T_C_K[~i] = float('nan')
    T_S_K[~i] = float('nan')

    return np.asarray(T_C_K, dtype=np.float32), np.asarray(T_S_K, dtype=np.float32)


def calc_T_CS_4SAIL(
        LAI,
        lidf,
        hotspot,
        Eo_n,
        Eo_f,
        L_sky,
        sza_n,
        sza_f,
        vza_n,
        vza_f,
        psi_n,
        psi_f,
        e_v,
        e_s):
    '''Estimates canopy and soil temperature by analytical inversion of 4SAIL
    (Eq. 12 in [Verhoef2007]_) of two directional radiometric observations. Ignoring shadows.

    Parameters
    ----------
    LAI : float
        Leaf (Plant) Area Index.
    lidf : list
        Campbell 1988 Leaf Inclination Distribution Function, default 5 degrees angle step.
    hotspot : float
        hotspot parameters, use 0 to ignore the hotspot effect (turbid medium).
    Eo_n : float
        Surface land Leaving thermal radiance (emitted thermal radiation).
        at the nadir observation (W m-2).
    Eo_f : float
        Surface land Leaving thermal radiance (emitted thermal radiation)
        at the oblique observation (W m-2).
    L_dn : float
        Broadband incoming longwave radiation (W m-2).
    sza_n : float
        Sun Zenith Angle during the nadir observation (degrees).
    sza_f : float
        Sun Zenith Angle during the oblique observation (degrees).
    vza_n : float
        View Zenith Angle during the nadir observation (degrees).
    vza_f : float
        View Zenith Angle during the oblique observation (degrees).
    psi_n : float
        Relative (sensor-sun) Azimuth Angle during the nadir observation (degrees).
    psi_f : float
        Relative (sensor-sun) Azimuth Angle during the oblique observation (degrees).
    e_v : float
        broadband leaf emissivity.
    e_s : float
        broadband soil emissivity.

    Returns
    -------
    T_C_K : float
        Canopy temperature (K).
    T_S_K : float
        Soil temperature (K).

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007)
    '''

    # Apply Kirkchoff to work with reflectances instead of emissivities
    r_s = 1. - e_s
    r_v = 1. - e_v
    # Get nadir parameters for the inversion
    [rdot_star_n,
     emiss_v_eff_n,
     emiss_s_eff_n,
     gamma_sot,
     emiss_sot] = calc_4SAIL_emission_param(LAI,
                                            hotspot,
                                            lidf,
                                            sza_n,
                                            vza_n,
                                            psi_n,
                                            r_v,
                                            r_s)
    # Calculate the total emission of the surface at nadir observation
    L_emiss_n = Eo_n - rdot_star_n * L_sky
    # Get forward parameters for the inversion
    [rdot_star_f,
     emiss_v_eff_f,
     emiss_s_eff_f,
     gamma_sot,
     emiss_sot] = calc_4SAIL_emission_param(LAI,
                                            hotspot,
                                            lidf,
                                            sza_f,
                                            vza_f,
                                            psi_f,
                                            r_v,
                                            r_s)
    # Calculate the total emission of the surface at oblique observation
    L_emiss_f = Eo_f - rdot_star_f * L_sky
    # Invert 4SAIL to get the BB emission of vegetation and soil
    H_v = ((emiss_s_eff_n * L_emiss_f - emiss_s_eff_f * L_emiss_n)
           / (emiss_s_eff_n * emiss_v_eff_f - emiss_s_eff_f * emiss_v_eff_n))
    H_S = (L_emiss_n - emiss_v_eff_n * H_v) / emiss_s_eff_n
    # Invert Stephan Boltzmann to obtain vegetation and soil temperatures
    T_C_K = (H_v / SB)**0.25
    T_S_K = (H_S / SB)**0.25
    return np.asarray(T_C_K, dtype=np.float32), np.asarray(T_S_K, dtype=np.float32)


def calc_4SAIL_emission_param(
        LAI,
        hotspot,
        lidf,
        sza,
        vza,
        psi,
        rho_v,
        rho_s,
        tau_v=0.0):
    '''Calculates the effective surface reflectance, and emissivities for
    soil and canopy using 4SAIL.

    Parameters
    ----------
    LAI : float
        Leaf (Plant) Area Index.
    hotspot : float
        hotspot parameters, use 0 to ignore the hotspot effect (turbid medium).
    lidf : list
        Campbell 1988 Leaf Inclination Distribution Function, 5 angle step.
    sza : float
        Sun Zenith Angle during the nadir observation (degrees).
    vza : float
        View Zenith Angle during the nadir observation (degrees).
    psi : float
        Relative (sensor-sun) Azimuth Angle during the nadir observation (degrees).
    psi_f : float
        Relative (sensor-sun) Azimuth Angle during the oblique observation (degrees).
    rho_v : float
        leaf reflectance (1-leaf emissivity).
    rho_s : float
        soil emissivity (1-soil emissivity).
    tau_v : float
        leaf transmittance (default zero transmittance in the TIR).

    Returns
    -------
    rdot_star : float
        surface effective reflectance.
    emiss_v_eff : float
        canopy effective emissivity.
    emiss_s_eff : float
        soil effective emissivity.
    gamma_sot : float
        directional canopy absortivity.
    emiss_sot : float
        directional canopy emissivity.

    References
    ----------
    Equations 5, 11, and 13 in [Verhoef2007]_
    '''

    # Run 4 SAIL
    [tss,
     too,
     tsstoo,
     rdd,
     tdd,
     rsd,
     tsd,
     rdo,
     tdo,
     rso,
     rsos,
     rsod,
     rddt,
     rsdt,
     rdot,
     rsodt,
     rsost,
     rsot,
     gamma_sdf,
     gammas_db,
     gamma_so] = foursail(LAI,
                          hotspot,
                          lidf,
                          sza,
                          vza,
                          psi,
                          rho_v,
                          tau_v,
                          rho_s)
    # Eq. 5 in [Verhoef2007]_
    gamma_d = 1. - rdd - tdd
    gamma_o = 1. - rdo - tdo - too
    # Eq. 13 in [Verhoef2007]_
    dn = 1. - rddt - rdd
    emiss_o = 1. - rdot
    emiss_d = 1. - rddt
    rdot_star = rdo + (tdd * (rddt * tdo + rdot * too) / dn)
    # Get the coefficients from Eq 11 in [Verhoef2007]_
    # 2nd element in Eq. 11 [Verhoef2007]_
    emiss_v_eff = gamma_o + (gamma_d * (rddt * tdo + rdot * too) / dn)
    # 3rd element in Eq. 11 [Verhoef2007]_
    emiss_s_eff = emiss_o * too + (emiss_d * (tdo + rdd * rdot * too) / dn)
    # 4th element in Eq. 11 [Verhoef2007]_
    gamma_sot = gamma_so + (gamma_sdf * (rddt * tdo + rdot * too) / dn)
    # 5th element in Eq. 11 [Verhoef2007]_
    emiss_sot = emiss_o * tsstoo + tss * \
        (emiss_d * (tdo + rdd * rdot * too) / dn)

    rdot_star, emiss_v_eff, emiss_s_eff, gamma_sot, emiss_sot = map(
        np.asarray, (rdot_star, emiss_v_eff, emiss_s_eff, gamma_sot, emiss_sot))
    return rdot_star, emiss_v_eff, emiss_s_eff, gamma_sot, emiss_sot


def calc_T_S(T_R, T_C, f_theta):
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

    # Convert the input scalars to numpy arrays
    T_R, T_C, f_theta = map(np.asarray, (T_R, T_C, f_theta))
    T_temp = T_R**4 - f_theta * T_C**4
    T_S = np.zeros(T_R.shape, np.float32)
    flag = np.zeros(T_R.shape, np.int32) + F_ALL_FLUXES

    # Succesfull inversion
    T_S[T_temp >= 0] = (T_temp[T_temp >= 0]
                        / (1.0 - f_theta[T_temp >= 0]))**0.25

    # Unsuccesfull inversion
    T_S[np.logical_or(T_temp < 0, np.isnan(T_temp))] = 1e-6
    flag[np.logical_or(T_temp < 0, np.isnan(T_temp))] = F_INVALID

    return np.asarray(flag, dtype=np.int32), np.asarray(T_S, dtype=np.float32)


def calc_T_S_4SAIL(T_R, T_C, rdot_star, emiss_v_eff, emiss_s_eff, L_dn=0):
    '''Estimates canopy temperature from the directional LST using 4SAIL parameters.

    Parameters
    ----------
    T_R : float
        Directional Radiometric Temperature (K)
    T_S : float
        Soil Temperature (K)
    rdot_star : float
        surface effective reflectance
    emiss_v_eff : float
        canopy effective emissivity
    emiss_s_eff : float
        soil effective emissivity
    L_dn : float
        downwelling atmospheric longwave radiance (W m-2)

    Returns
    -------
    flag : int
        Error flag if inversion not possible (255).
    T_S : float
        Soil temperature (K).'''

    Hv = met.calc_stephan_boltzmann(T_C)
    Eo = met.calc_stephan_boltzmann(T_R)
    Hs = np.asarray((Eo - rdot_star * L_dn - emiss_v_eff * Hv) / emiss_s_eff)

    flag = np.zeros(T_R.shape) + F_ALL_FLUXES
    T_S = np.zeros(T_R.shape)

    # Succesfull inversion
    T_S[Hs >= 0] = (Hs / SB)**0.25

    # Unsuccesfull inversion
    T_S[Hs < 0] = 1e-6
    flag[Hs < 0] = F_INVALID

    return np.asarray(flag), np.asarray(T_S, dtype=np.float32)


def calc_T_S_series(Tr_K, T_A_K, R_A, R_x, R_S, f_theta, H_S, rho, c_p):
    '''Estimates soil temperature from soil sensible heat flux and
    resistance network in series.

    Parameters
    ----------
    Tr_K : float
        Directional Radiometric Temperature (K).
    T_A_K : float
        Air Temperature (K).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk aerodynamic resistance to heat transport at the canopy boundary layer (s m-1).
    R_S : float
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
    T_S: float
        Soil temperature (K).
    T_C : float
        Air temperature at the canopy interface (K).

    References
    ----------
    Eqs. A15-A19 from [Norman1995]_'''

    # Eq. A.15 Norman 1995
    T_AC_lin = (((T_A_K / R_A) + (Tr_K / (f_theta * R_x))
                - (((1.0 - f_theta) / (f_theta * R_x)) * H_S * R_S / (rho * c_p))
                + H_S / (rho * c_p))
                / ((1.0 / R_A) + (1.0 / R_x) + (1.0 - f_theta) / (f_theta * R_x)))
    # Eq. A.17 Norman 1995
    T_e = T_AC_lin * (1.0 + (R_x / R_A)) - H_S * R_x / \
        (rho * c_p) - T_A_K * R_x / R_A
    # Eq. A.16 Norman 1995
    Delta_T_AC = ((Tr_K**4 - (1.0 - f_theta) * (H_S * R_S / (rho * c_p) + T_AC_lin)**4
                  - f_theta * T_e**4) / (4 * f_theta * T_e**3.0 * (1.0 + (R_x / R_A))
                  + 4.0 * (1.0 - f_theta) * (H_S * R_S / (rho * c_p) + T_AC_lin)**3))
    # Eq. A.18 Norman 1995
    T_AC = T_AC_lin + Delta_T_AC
    T_S = T_AC + H_S * R_S / (rho * c_p)
    return np.asarray(T_S, dtype=np.float32), np.asarray(T_AC, dtype=np.float32)


def _check_default_parameter_size(parameter, input_array):

    parameter = np.asarray(parameter, dtype=np.float32)
    if parameter.size == 1:
        parameter = np.ones(input_array.shape) * parameter
        return np.asarray(parameter, dtype=np.float32)
    elif parameter.shape != input_array.shape:
        raise ValueError(
            'dimension mismatch between parameter array and input array with shapes %s and %s' %
            (parameter.shape, input_array.shape))
    else:
        return np.asarray(parameter, dtype=np.float32)


def calc_resistances(res_form, res_types):
    '''Calculate the aerodynamic resistances: R_A, R_x and R_S.

    Parameters
    ----------
    res_form : int
        Constant specifying which resistance formulation to use:
        KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1),
        MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)
        If the constant is not any of the above then KUSTAS_NORMAN_1999 is
        used.
    res_types : Dictionary of dictionaries
        Dictionary specifying which of the three resistances to calculate. For
        each resistance to calculate the dictionary must contain a key-value
        pair with the key being the name of the resistance and value being
        another dictionary with all the parameters required to calculate the
        given resistance.
        Key: R_A
        R_A Parameters: 'z_T', 'u_friction', 'L', 'd_0', 'z_0H'
        Key: R_x
        R_x Parameters: 'u_friction', 'h_C', 'd_0', 'z_0M', 'L', 'F', 'LAI',
                        'leaf_width', 'res_params'
        Key: R_S
        R_S Parameters: 'u_friction', 'h_C', 'd_0', 'z_0M', 'L', 'omega0', 'F',
                        'leaf_width', 'z0_soil', 'z_u', 'deltaT', 'res_params'

    Returns
    -------
    R_A: float array or None
        Aerodyamic resistance to heat transport in the surface layer (s m-1)
    R_x : float array or None
        Aerodynamic resistance at the canopy boundary layer (s m-1)
    R_S: float array or None
        Aerodynamic resistance at the  soil boundary layer (s m-1)

    '''

    R_A = 0
    R_x = 0
    R_S = 0
    u_C = None

    if res_form not in [KUSTAS_NORMAN_1999, CHOUDHURY_MONTEITH_1988,
                        MCNAUGHTON_VANDERHURK, CHOUDHURY_MONTEITH_ALPHA_1988,
                        HADHIGHI_AND_OR_2015]:
        res_form = KUSTAS_NORMAN_1999

    # Determine which resistances to calculate and get the required parameters
    if 'R_A' in res_types.keys():
        z_T, u_friction, L, d_0, z_0H = [res_types['R_A'].get(k)
                                         for k in ['z_T',
                                                   'u_friction',
                                                   'L',
                                                   'd_0',
                                                   'z_0H']]
        del res_types['R_A']
        calc_R_A = True
    else:
        calc_R_A = False
    if 'R_x' in res_types.keys():
        u_friction, h_C, d_0, z_0M, L, F, LAI, leaf_width, z0_soil, massman_profile, res_params = \
            [res_types['R_x'].get(k) for k in ['u_friction', 'h_C', 'd_0', 'z_0M',
                                               'L', 'F', 'LAI', 'leaf_width',
                                               'z0_soil', 'massman_profile',
                                               'res_params']]

        del res_types['R_x']
        calc_R_x = True
    else:
        calc_R_x = False
    if 'R_S' in res_types.keys():
        u_friction, h_C, d_0, z_0M, L, omega0, F, leaf_width, z0_soil, z_u, deltaT, u, rho,\
         c_p, f_cover, w_C, res_params, LAI, massman_profile = \
             [res_types['R_S'].get(k) for k in ['u_friction', 'h_C', 'd_0', 'z_0M',
                                                'L', 'omega0', 'F', 'leaf_width',
                                                'z0_soil', 'z_u', 'deltaT',
                                                'u', 'rho', 'c_p', 'f_cover',
                                                'w_C', 'res_params', "LAI",
                                                "massman_profile"]]

        del res_types['R_S']
        calc_R_S = True
    else:
        calc_R_S = False

    # Calculate the aerodynamic resistance
    if calc_R_A:
        R_A = res.calc_R_A(z_T, u_friction, L, d_0, z_0H)
        del z_T, z_0H

    # Calculate soil and canopy resistances
    if res_form == KUSTAS_NORMAN_1999:
        if calc_R_x:
            u_C = wnd.calc_u_C_star(u_friction, h_C, d_0, z_0M, L)
            u_C = np.maximum(u_C, U_C_MIN)
            # Wind speed is highly attenuated within the canopy volume
            if massman_profile[0] <= 0:
                u_d_zm = wnd.calc_u_Goudriaan(u_C, h_C, F, leaf_width, d_0+z_0M)
            else:
                u_d_zm = wnd.calc_u_Massman(u_C, h_C, F, d_0+z_0M,
                                            massman_profile[1],
                                            xi_soil=z0_soil/h_C,
                                            c_d=massman_profile[0])

            u_d_zm = np.maximum(u_d_zm, U_C_MIN)
            # Vegetation in series with soil, i.e. well mixed, so we use
            # the landscape LAI
            R_x = res.calc_R_x_Norman(LAI, leaf_width, u_d_zm, res_params)
            del u_d_zm
        if calc_R_S:
            if u_C is None:
                u_C = wnd.calc_u_C_star(u_friction, h_C, d_0, z_0M, L)
                u_C = np.maximum(u_C, U_C_MIN)
            # Clumped vegetation enhanced wind speed for the soil surface
            if massman_profile[0] <= 0:
                u_S = wnd.calc_u_Goudriaan(u_C, h_C, LAI, leaf_width, z0_soil)
            else:
                u_S = wnd.calc_u_Massman(u_C, h_C, LAI, z0_soil,
                                         massman_profile[1],
                                         xi_soil=z0_soil/h_C,
                                         c_d=massman_profile[0])
            u_S = np.maximum(u_S, U_S_MIN)
            R_S = res.calc_R_S_Kustas(u_S, deltaT, params=res_params)

    elif res_form == CHOUDHURY_MONTEITH_1988:
        if calc_R_x:
            u_C = wnd.calc_u_C_star(u_friction, h_C, d_0, z_0M, L)
            u_C = np.maximum(u_C, U_C_MIN)
            # Vegetation in series with soil, i.e. well mixed, so we use
            # the landscape LAI
            R_x = res.calc_R_x_Choudhury(u_C, LAI, leaf_width)
            del LAI, leaf_width
        if calc_R_S:
            R_S = res.calc_R_S_Choudhury(u_friction, h_C, z_0M, d_0, z_u, z0_soil)

    elif res_form == MCNAUGHTON_VANDERHURK:
        if calc_R_x:
            # Vegetation in series with soil, i.e. well mixed, so we use
            # the landscape LAI
            R_x = res.calc_R_x_McNaughton(LAI, leaf_width, u_friction)
            del LAI, leaf_width
        if calc_R_S:
            R_S = res.calc_R_S_McNaughton(u_friction)

    elif res_form == CHOUDHURY_MONTEITH_ALPHA_1988:
        if calc_R_x:
            u_C = wnd.calc_u_C_star(u_friction, h_C, d_0, z_0M, L)
            u_C = np.maximum(u_C, U_C_MIN)
            # Wind speed is highly attenuated within the canopy volume
            alpha_prime = wnd.calc_A_Goudriaan(h_C, LAI, leaf_width)
            # Vegetation in series with soil, i.e. well mixed, so we use
            # the landscape LAI
            R_x = res.calc_R_x_Choudhury(u_C, LAI, leaf_width, alpha_prime=alpha_prime)
            del LAI, alpha_prime

        if calc_R_S:
            # Clumped vegetation enhanced wind speed for the soil surface
            alpha_k = wnd.calc_A_Goudriaan(h_C, LAI, leaf_width)
            R_S = res.calc_R_S_Choudhury(u_friction, h_C, z_0M, d_0, z_u, z0_soil, alpha_k=alpha_k)

    elif res_form == HADHIGHI_AND_OR_2015:
        if calc_R_x:
            u_C = wnd.calc_u_C_star(u_friction, h_C, d_0, z_0M, L)
            u_C = np.maximum(u_C, U_C_MIN)
            # Wind speed is highly attenuated within the canopy volume
            if massman_profile[0] <= 0:
                u_d_zm = wnd.calc_u_Goudriaan(u_C, h_C, F, leaf_width, d_0+z_0M)
            else:
                u_d_zm = wnd.calc_u_Massman(u_C, h_C, F, d_0+z_0M,
                                            massman_profile[1],
                                            xi_soil=z0_soil/h_C,
                                            c_d=massman_profile[0])

            u_d_zm = np.maximum(u_d_zm, U_C_MIN)
            # Vegetation in series with soil, i.e. well mixed, so we use
            # the landscape LAI
            R_x = res.calc_R_x_Norman(LAI, leaf_width, u_d_zm, res_params)
            del LAI, leaf_width, u_d_zm
        if calc_R_S:
            # Wind speed is highly attenuated within the canopy volume
            if massman_profile[0] <= 0:
                u_star_soil = None
            else:
                u_star_ratio_2 = wnd.calc_ustar_massman(h_C, F, z0_soil,
                                                      massman_profile[1],
                                                      Xi_soil=z0_soil/h_C,
                                                      C_d=massman_profile[0])

                u_star_soil = u_friction * np.sqrt(u_star_ratio_2)


            R_S = res.calc_R_S_Haghighi(u, h_C, z_u, rho, c_p,
                                        z0_soil=z0_soil,
                                        f_cover=f_cover,
                                        w_C=w_C)

    R_A = np.asarray(np.clip(R_A, R_A_MIN, R_A_MAX))
    R_x = np.asarray(np.clip(R_x, RES_MIN, RES_MAX))
    R_S = np.asarray(np.clip(R_S, RES_MIN, RES_MAX))

    return R_A, R_x, R_S



def monin_obukhov_convergence(l_mo, l_queue, l_converged, flag):
    l_new = np.array(l_mo)
    l_new[l_new == 0] = 1e-36
    l_queue.appendleft(l_new)
    i = np.logical_and(~l_converged, flag != F_INVALID)
    if np.sum(i) <= 1 or np.size(i) == 0:
        return i, l_queue, l_converged, np.inf


    if len(l_queue) >= 4:
        i = np.logical_and(~l_converged, flag != F_INVALID)
        if np.any(i):
            l_converged[i] = np.logical_and(
                _L_diff(l_queue[0][i], l_queue[2][i]) < L_thres,
                _L_diff(l_queue[1][i], l_queue[3][i]) < L_thres)

    if len(l_queue) == 6:
        i = np.logical_and(~l_converged, flag != F_INVALID)
        if np.any(i):
            l_converged[i] = np.logical_and.reduce(
                (_L_diff(l_queue[0][i], l_queue[3][i]) < L_thres,
                 _L_diff(l_queue[1][i], l_queue[4][i]) < L_thres,
                 _L_diff(l_queue[2][i], l_queue[5][i]) < L_thres))

    if np.sum(i) == 0 or np.size(i) == 0:
        return i, l_queue, l_converged, np.inf

    l_diff_max = np.nanmax(_L_diff(l_queue[0][i], l_queue[1][i]))
    
    return i, l_queue, l_converged, l_diff_max
    
    
