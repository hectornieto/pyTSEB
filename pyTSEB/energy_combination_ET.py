# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:58:34 2017

@author: hnieto
"""
from collections import deque
import time

from pyTSEB import TSEB
import numpy as np

# kB coefficient
kB = 0

ITERATIONS = 15

TALL_REFERENCE = 1
SHORT_REFERENCE = 0


def penman_monteith(T_A_K, u, ea, p, Sn, L_dn, emis, LAI, z_0M, d_0, z_u, z_T,
                    calcG_params=None, const_L=None,
                    Rst_min=100,
                    leaf_type=TSEB.res.AMPHISTOMATOUS,
                    environmental_factors=1):

    """Penman Monteith [Allen1998]_ energy combination model.
    Calculates the Penman Monteith one source fluxes using meteorological and crop data.

    Parameters
    ----------
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn : float
        Net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    emis : float
        Surface emissivity.
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
        Height of measurement of air temperature (m).
    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with
                                                       G_param list of parameters
                                                       (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    Rst_min : float
        Minimum (unstress) single-leaf stomatal coductance (s m -1), Default = 100 s m-1
    leaf_type : int
        1: Hipostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)
    environmental_factors : float [0-1]
        Correction factor for stomatal conductance in case of biotic (water)
        or abiotic (atmospheric) stress. Default = 1.

    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    L_n : float
        Net longwave radiation (W m-2)
    LE : float
        Latent heat flux (W m-2).
    H : float
        Sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
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
    .. [Allen1998] R.G. Allen, L.S. Pereira, D. Raes, M. Smith. Crop
        Evapotranspiration (guidelines for computing crop water requirements),
        FAO Irrigation and Drainage Paper No. 56. 1998
    """

    if calcG_params is None:
        calcG_params = [[1], 0.35]

    # Convert float scalars into numpy arrays and check parameters size
    T_A_K = np.asarray(T_A_K)
    [u,
     ea,
     p,
     Sn,
     L_dn,
     emis,
     LAI,
     z_0M,
     d_0,
     z_u,
     z_T,
     calcG_array,
     Rst_min,
     leaf_type] = map(
         TSEB._check_default_parameter_size,
         [u, ea, p, Sn, L_dn, emis, LAI, z_0M, d_0, z_u, z_T,
          calcG_params[1], Rst_min, leaf_type], [T_A_K] * 14)

    # Create the output variables
    [flag, Ln, LE, H, G, R_A, iterations] = [np.zeros(T_A_K.shape) + np.NaN for i in range(7)]

    # Calculate the general parameters
    rho_a = TSEB.met.calc_rho(p, ea, T_A_K)  # Air density
    Cp = TSEB.met.calc_c_p(p, ea)            # Heat capacity of air

    # slope of saturation water vapour pressure in mb K-1
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(T_A_K)

    lambda_ = TSEB.met.calc_lambda(T_A_K)        # latent heat of vaporization MJ kg-1
    psicr = TSEB.met.calc_psicr(Cp, p, lambda_)  # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(T_A_K)     # saturation water vapour pressure in mb

    rho_cp = rho_a * Cp
    vpd = es - ea

    # Calculate bulk stomatal conductance
    R_c = bulk_stomatal_conductance(LAI,
                                    Rst_min,
                                    leaf_type=leaf_type,
                                    environmental_factors=1)

    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(T_A_K.shape) + np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(T_A_K.shape) * const_L)
        max_iterations = 1  # No iteration
    u_friction = TSEB.MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(TSEB.u_friction_min, u_friction))

    L_old = np.ones(T_A_K.shape)
    L_diff = np.asarray(np.ones(T_A_K.shape) * np.inf)
    z_0H = TSEB.res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport

    # Calculate Net radiation
    T_0_K = T_A_K
    # asumme aerodynamic tempearture equals air temperature
    Ln = emis * (L_dn - TSEB.met.calc_stephan_boltzmann(T_0_K))
    Rn = np.asarray(Sn + Ln)

    # Compute Soil Heat Flux
    i = np.ones(Rn.shape, dtype=bool)
    G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn, i)

    for n_iterations in range(max_iterations):
        if np.all(L_diff < TSEB.L_thres):
            #print("Finished interation with a max. L diff: " + str(np.max(L_diff)))
            break

        # print("Iteration " + str(n_iterations) + , max. L diff: " + str(np.max(L_diff)))

        i = np.logical_and(L_diff >= TSEB.L_thres, flag != 255)
        iterations[i] = n_iterations
        flag[i] = 0

        # Calculate aerodynamic resistances
        R_A[i] = TSEB.res.calc_R_A(z_T[i], u_friction[i], L[i], d_0[i], z_0H[i])
        R_eff = R_c[i] / R_A[i]

        # Apply Penman Monteith Combination equation
        LE[i] = ((delta[i] * (Rn[i] - G[i]) + rho_cp[i] * vpd[i] / R_A[i])
                 / (delta[i] + psicr[i] * (1. + R_eff)))
        H[i] = Rn[i] - G[i] - LE[i]
        # Now L can be recalculated and the difference between iterations
        # derived
        if const_L is None:
            L[i] = TSEB.MO.calc_L(
                u_friction[i],
                T_A_K[i],
                rho_a[i],
                Cp[i],
                H[i],
                LE[i])
            L_diff = np.asarray(np.fabs(L - L_old) / np.fabs(L_old))
            L_diff[np.isnan(L_diff)] = np.inf
            L_old = np.array(L)
            L_old[L_old == 0] = 1e-36

            # Calculate again the friction velocity with the new stability
            # correctios
            u_friction[i] = TSEB.MO.calc_u_star(u[i], z_u[i], L[i], d_0[i], z_0M[i])
            u_friction = np.asarray(np.maximum(TSEB.u_friction_min, u_friction))

    flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations = map(
        np.asarray, (flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations))

    return flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations


def shuttleworth_wallace(T_A_K,
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
                         x_LAD=1,
                         f_c=1,
                         w_C=1,
                         Rst_min=100,
                         R_ss=500,
                         resistance_form=None,
                         calcG_params=None,
                         const_L=None,
                         massman_profile=None,
                         leaf_type=TSEB.res.AMPHISTOMATOUS,
                         environmental_factors=1):

    """Shuttleworth and Wallace [Shuttleworth1995]_ dual source energy combination model.
    Calculates turbulent fluxes using meteorological and crop data for a
    dual source system in series.

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
    x_LAD : float, optional
        Campbell 1990 leaf inclination distribution function chi parameter.
    f_c : float, optional
        Fractional cover.
    w_C : float, optional
        Canopy width to height ratio.
    Rst_min : float
        Minimum (unstress) single-leaf stomatal coductance (s m -1),
        Default = 100 s m-1
    Rss : float
        Resistance to water vapour transport in the soil surface (s m-1),
        Default = 500 s m-1 (moderately dry soil)
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.

            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.
            * 4 : Haghighi and Orr 2015

    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.

            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with
                                                       G_param list of parameters
                                                       (see :func:`~TSEB.calc_G_time_diff`).

    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    leaf_type : int
        1: Hipostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)
    environmental_factors : float [0-1]
        Correction factor for stomatal conductance in case of biotic (water)
        or abiotic (atmospheric) stress. Default = 1.

    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    vpd_0 : float
        Water pressure deficit at the canopy interface (mb).
    L_nS : float
        Soil net longwave radiation (W m-2)
    L_nC : float
        Canopy net longwave radiation (W m-2)
    LE : float
        Latent heat flux (W m-2).
    H : float
        Sensible heat flux (W m-2).
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
    .. [Shuttleworth1995] W.J. Shuttleworth, J.S. Wallace, Evaporation from
        sparse crops - an energy combinatino theory,
        Quarterly Journal of the Royal Meteorological Society , Volume 111, Issue 469,
        Pages 839-855,
        http://dx.doi.org/10.1002/qj.49711146910.
    """

    if resistance_form is None:
        resistance_form = [0, {}]

    if calcG_params is None:
        calcG_params = [[1], 0.35]

    if massman_profile is None:
        massman_profile = [0, []]

    # Convert float scalars into numpy arrays and check parameters size
    T_A_K = np.asarray(T_A_K)
    [u,
     ea,
     p,
     Sn_C,
     Sn_S,
     L_dn,
     LAI,
     emis_C,
     emis_S,
     h_C,
     z_0M,
     d_0,
     z_u,
     z_T,
     leaf_width,
     z0_soil,
     x_LAD,
     f_c,
     w_C,
     Rst_min,
     R_ss,
     calcG_array,
     leaf_type] = map(
         TSEB._check_default_parameter_size,
         [u,
          ea,
          p,
          Sn_C,
          Sn_S,
          L_dn,
          LAI,
          emis_C,
          emis_S,
          h_C,
          z_0M,
          d_0,
          z_u,
          z_T,
          leaf_width,
          z0_soil,
          x_LAD,
          f_c,
          w_C,
          Rst_min,
          R_ss,
          calcG_params[1],
          leaf_type],
         [T_A_K] * 24)

    res_params = resistance_form[1]
    resistance_form = resistance_form[0]

    # Create the output variables
    [flag, vpd_0, LE, H, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A,
     iterations] = [np.zeros(T_A_K.shape) + np.NaN for i in range(13)]

    # Calculate the general parameters
    rho_a = TSEB.met.calc_rho(p, ea, T_A_K)              # Air density
    Cp = TSEB.met.calc_c_p(p, ea)                        # Heat capacity of air

    # slope of saturation water vapour pressure in mb K-1
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(T_A_K)

    lambda_ = TSEB.met.calc_lambda(T_A_K)                     # latent heat of vaporization MJ kg-1
    psicr = TSEB.met.calc_psicr(Cp, p, lambda_)                     # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(T_A_K)             # saturation water vapour pressure in mb

    rho_cp = rho_a * Cp
    vpd = es - ea

    # Calculate bulk stomatal conductance
    R_c = bulk_stomatal_conductance(LAI, Rst_min, leaf_type=leaf_type,
                                    environmental_factors=environmental_factors)

    F = np.asarray(LAI / f_c)  # Real LAI
    omega0 = TSEB.CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)

    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(T_A_K.shape) + np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(T_A_K.shape) * const_L)
        max_iterations = 1  # No iteration
    u_friction = TSEB.MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(TSEB.u_friction_min, u_friction))
    L_queue = deque([np.array(L)], 6)
    L_converged = np.asarray(np.zeros(T_A_K.shape)).astype(bool)
    L_diff_max = np.inf
 
    z_0H = TSEB.res.calc_z_0H(z_0M, kB=0)  # Roughness length for heat transport
   
    # First assume that temperatures equals the Air Temperature
    T_C, T_S = np.array(T_A_K), np.array(T_A_K)
    Ln_C, Ln_S = TSEB.rad.calc_L_n_Kustas(T_C, T_S, L_dn, LAI, emis_C, emis_S, x_LAD=x_LAD)

    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = ~L_converged
        if np.all(L_converged):
            if L_converged.size == 0:
                print("Finished iterations with no valid solution")
            else:
                print("Finished interations with a max. L diff: " + str(L_diff_max))
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        print("Iteration: %d, non-converged pixels: %d, max L diff: %f, total time: %f, loop time: %f" %
              (n_iterations, np.sum(i), L_diff_max, total_duration, loop_duration))
        
        iterations[i] = n_iterations
        flag[i] = 0

        # Calculate aerodynamic resistances
        R_A[i], R_x[i], R_S[i] = TSEB.calc_resistances(resistance_form,
                                                       {"R_A": {"z_T": z_T[i], "u_friction": u_friction[i], "L": L[i],
                                                          "d_0": d_0[i], "z_0H": z_0H[i]},
                                                        "R_x": {"u_friction": u_friction[i], "h_C": h_C[i],
                                                               "d_0": d_0[i],
                                                          "z_0M": z_0M[i], "L": L[i],  "LAI": LAI[i],
                                                          "leaf_width": leaf_width[i], "massman_profile": massman_profile,
                                                          "res_params": {k: res_params[k][i] for k in res_params.keys()}},
                                                        "R_S": {"u_friction": u_friction[i], 'u':u[i], "h_C": h_C[i], "d_0": d_0[i],
                                                          "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i],
                                                           "LAI": LAI[i], "leaf_width": leaf_width[i],
                                                           "z0_soil": z0_soil[i], "z_u": z_u[i],
                                                           "deltaT": T_S[i] - T_C[i], "massman_profile": massman_profile,
                                                           'rho':rho_a[i], 'c_p':Cp[i], 'f_cover':f_c[i],
                                                           'w_C':w_C[i],
                                                           "res_params": {k: res_params[k][i] for k in res_params.keys()}}
                                                        }
                                                       )

        _, _, _, C_s, C_c = calc_effective_resistances_SW(R_A[i],
                                               R_x[i], 
                                               R_S[i], 
                                               R_c[i],
                                               R_ss[i],
                                               delta[i],
                                               psicr[i])

        Rn_C = Sn_C + Ln_C
        Rn_S = Sn_S + Ln_S
        Rn = Rn_C + Rn_S
        # Compute Soil Heat Flux Ratio
        G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn_S, i)

        # Eq. 12 in [Shuttleworth1988]_
        PM_C = ((delta[i] * (Rn[i] - G[i]) + (rho_cp[i] * vpd[i] - delta[i] * R_x[i]
                * (Rn_S[i] - G[i])) / (R_A[i] + R_x[i]))
                / (delta[i] + psicr[i] * (1. + R_c[i] / (R_A[i] + R_x[i]))))
        # Eq. 13 in [Shuttleworth1988]_
        PM_S = ((delta[i] * (Rn[i] - G[i]) + (rho_cp[i] * vpd[i] - delta[i] * R_S[i] * Rn_C[i])
                / (R_A[i] + R_S[i])) / (delta[i] + psicr[i] * (1. + R_ss[i] / (R_A[i] + R_S[i]))))
        # Eq. 11 in [Shuttleworth1988]_
        LE[i] = C_c * PM_C + C_s * PM_S
        H[i] = Rn[i] - G[i] - LE[i]

        # Compute canopy and soil  fluxes
        # Vapor pressure deficit at canopy source height (mb) # Eq. 8 in [Shuttleworth1988]_
        vpd_0[i] = (vpd[i] + (delta[i] * (Rn[i] - G[i])
                    - (delta[i] + psicr[i]) * LE[i]) * R_A[i] / (rho_cp[i]))
        # Eq. 9 in Shuttleworth & Wallace 1985
        LE_S[i] = ((delta[i] * (Rn_S[i] - G[i]) + rho_cp[i] * vpd_0[i] / R_S[i])
                   / (delta[i] + psicr[i] * (1. + R_ss[i] / R_S[i])))
        H_S[i] = Rn_S[i] - G[i] - LE_S[i]
        # Eq. 10 in Shuttleworth & Wallace 1985
        LE_C[i] = ((delta[i] * Rn_C[i] + rho_cp[i] * vpd_0[i] / R_x[i])
                   / (delta[i] + psicr[i] * (1. + R_c[i] / R_x[i])))
        H_C[i] = Rn_C[i] - LE_C[i]

        T_C[i] = calc_T(H_C[i], T_A_K[i], R_A[i] + R_x[i], rho_a[i], Cp[i])
        T_S[i] = calc_T(H_S[i], T_A_K[i], R_A[i] + R_S[i], rho_a[i], Cp[i])
        no_valid_T = np.logical_and(i, T_C < 0)
        flag[no_valid_T] = TSEB.F_INVALID
        T_C[no_valid_T] = T_A_K[no_valid_T]
        no_valid_T = np.logical_and(i, T_S < 0)
        flag[no_valid_T] = TSEB.F_INVALID
        T_S[no_valid_T] = T_A_K[no_valid_T]
        # Calculate net longwave radiation with current values of T_C and T_S
        Ln_C[i], Ln_S[i] = TSEB.rad.calc_L_n_Kustas(T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i], x_LAD=x_LAD[i])

        # Now L can be recalculated and the difference between iterations
        # derived
        if const_L is None:
            L[i] = TSEB.MO.calc_L(
                    u_friction[i],
                    T_A_K[i],
                    rho_a[i],
                    Cp[i],
                    H[i],
                    LE[i])

            # Calculate again the friction velocity with the new stability
            # correctios
            u_friction[i] = TSEB.MO.calc_u_star(
                u[i], z_u[i], L[i], d_0[i], z_0M[i])
            u_friction[i] = np.asarray(np.maximum(TSEB.u_friction_min, u_friction[i]))
            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            L_new = np.array(L)
            L_new[L_new == 0] = 1e-36
            L_queue.appendleft(L_new)
            L_converged[i] = TSEB._L_diff(L_queue[0][i], L_queue[1][i]) < TSEB.L_thres
            L_diff_max = np.max(TSEB._L_diff(L_queue[0][i], L_queue[1][i]))
            if len(L_queue) >= 4:
                L_converged[i] = np.logical_and(TSEB._L_diff(L_queue[0][i], L_queue[2][i]) < TSEB.L_thres,
                                                TSEB._L_diff(L_queue[1][i], L_queue[3][i]) < TSEB.L_thres)
            if len(L_queue) == 6:
                L_converged[i] = np.logical_and.reduce((TSEB._L_diff(L_queue[0][i], L_queue[3][i]) < TSEB.L_thres,
                                                        TSEB._L_diff(L_queue[1][i], L_queue[4][i]) < TSEB.L_thres,
                                                        TSEB._L_diff(L_queue[2][i], L_queue[5][i]) < TSEB.L_thres))    
    (flag,
     T_S,
     T_C,
     vpd_0,
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
                          vpd_0,
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

    return (flag, T_S, T_C, vpd_0, Ln_S, Ln_C, LE, H, LE_C, H_C,
            LE_S, H_S, G, R_S, R_x, R_A, u_friction, L, n_iterations)


def pet_asce(T_A_K, u, ea, p, Sdn, z_u, z_T, f_cd=1, reference=TALL_REFERENCE):
    '''Calcultaes the latent heat flux for well irrigated and cold pixel using
    ASCE potential ET from a tall (alfalfa) crop
    Parameters
    ----------
    T_A_K : float or array
        Air temperature (Kelvin).
    u : float or array
        Wind speed above the canopy (m s-1).
    ea : float or array
        Water vapour pressure above the canopy (mb).
    p : float or array
        Atmospheric pressure (mb), use 1013 mb by default.
    Sdn : float or array
        Solar irradiance (W m-2).
    z_u : float or array
        Height of measurement of windspeed (m).
    z_T : float or array
        Height of measurement of air temperature (m).
    f_cd : float or array
        cloudiness factor, default = 1
    reference : bool
        If true, reference ET is for a tall canopy (i.e. alfalfa)
    Returns
    -------
    LE : float or array
        Potential latent heat flux (W m-2)
    '''
    # Atmospheric constants
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_ = TSEB.met.calc_lambda(T_A_K)                     # latent heat of vaporization MJ kg-1
    c_p = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air
    psicr = TSEB.met.calc_psicr(c_p, p, lambda_)                     # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(T_A_K)             # saturation water vapour pressure in mb

    # Net shortwave radiation
    # Sdn = Sdn * 3600 / 1e6 # W m-2 to MJ m-2 h-1
    albedo = 0.23
    Sn = Sdn * (1.0 - albedo)
    # Net longwave radiation
    Ln = calc_Ln(T_A_K, ea, f_cd=f_cd)
    # Net radiation
    Rn = Sn + Ln
    # Soil heat flux
    if reference == TALL_REFERENCE:
        G_ratio = 0.04
        h_c = 0.5
        C_d = 0.25
        C_n = 66.0
        # R_s = 30.0
    else:
        G_ratio = 0.1
        h_c = 0.12
        C_d = 0.24
        C_n = 37.0
        # R_s = 50.0

    # Soil heat flux
    G = G_ratio * Rn
    # Windspeed at 2m height
    z_0M = h_c * 0.123
    d = h_c * 0.67
    u_2 = wind_profile(u, z_u, z_0M, d, 2.0)

    LE = (delta * (Rn - G) + psicr * C_n * u_2 * (es - ea) / T_A_K) / (delta + psicr * C_d * u_2)

    return LE


def pet_fao56(T_A_K, u, ea, p, Sdn, z_u, z_T, f_cd=1, reference=SHORT_REFERENCE):
    '''Calcultaes the latent heat flux for well irrigated and cold pixel using
    FAO56 potential ET from a short (grass) crop
    Parameters
    ----------
    T_A_K : float or array
        Air temperature (Kelvin).
    u : float or array
        Wind speed above the canopy (m s-1).
    ea : float or array
        Water vapour pressure above the canopy (mb).
    p : float or array
        Atmospheric pressure (mb), use 1013 mb by default.
    Sdn : float or array
        Solar irradiance (W m-2).
    z_u : float or array
        Height of measurement of windspeed (m).
    z_T : float or array
        Height of measurement of air temperature (m).
    f_cd : float or array
        cloudiness factor, default = 1
    reference : bool
        If true, reference ET is for a tall canopy (i.e. alfalfa)
    '''
    # Atmospheric constants
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_ = TSEB.met.calc_lambda(T_A_K)                     # latent heat of vaporization MJ kg-1
    c_p = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air
    psicr = TSEB.met.calc_psicr(c_p, p, lambda_)                     # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(T_A_K)             # saturation water vapour pressure in mb
    rho = TSEB.met.calc_rho(p, ea, T_A_K)

    # Net shortwave radiation
    # Sdn = Sdn * 3600 / 1e6 # W m-2 to MJ m-2 h-1
    albedo = 0.23
    Sn = Sdn * (1.0 - albedo)
    # Net longwave radiation
    Ln = calc_Ln(T_A_K, ea, f_cd=f_cd)
    # Net radiation
    Rn = Sn + Ln

    if reference == TALL_REFERENCE:
        G_ratio = 0.04
        h_c = 0.5
    else:
        G_ratio = 0.1
        h_c = 0.12

    R_c = 70.0

    # Soil heat flux
    G = G_ratio * Rn
    # Windspeed at 2m height
    z_0M = h_c * 0.123
    d = h_c * 2./3.
    u_2 = wind_profile(u, z_u, z_0M, d, 2.0)
    R_a = 208./u_2

    LE = (delta * (Rn - G) + rho * c_p * (es - ea) / R_a) / (delta + psicr * (1.0 + R_c / R_a))

    return LE


def bulk_stomatal_conductance(LAI, Rst, leaf_type=TSEB.res.AMPHISTOMATOUS, environmental_factors=1):
    """ Calculate the bulk canopy stomatal conductance.

    Parameters
    ----------
    LAI : float
        Leaf Area Index (m2 m-2).
    Rst_min : float
        Minimum (unstressed) single-leaf stomatal coductance (s m -1),
        Default = 100 s m-1
    leaf_type : int
        1: Hipostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)
    environmental_factors : float [0-1]
        Correction factor for stomatal conductance in case of biotic (water)
        or abiotic (atmospheric) stress. Default = 1.

    Returns
    -------
    R_c : float
        Canopy bulk stomatal conductance (s m-1)
    """

    R_c = Rst / (leaf_type * LAI * environmental_factors)
    return np.asarray(R_c)


def vpd_factor_Noilhan(T_A_K, ea, g_vpd=0.025):
    """ Estimate stomatal stress due to vapour pressure deficit based on [Noilhan]_

    Parameteers
    -----------
    T_A_K : float
        Air temperature (Kelvin).
    ea : float
        Water vapour pressure above the canopy (mb).
    g_vdp : float
        Empirical scale coefficient, use 0.025 mb-1 by default.

    Returns
    -------
    f : float
        Reduction factor in stomatal conductance [0-1]

    References
    ----------
    .. [Noilhan1989] J. Noilhan, S. Planton, A simple parameterization of
        land surface processes for meteorological models,
        Monthly Weather Review , Volume 117, 1989,
        Pages 536-549,
        https://doi.org/10.1175/1520-0493(1989)117<0536:ASPOLS>2.0.CO;2.
    """

    es = TSEB.met.calc_vapor_pressure(T_A_K)  # Calculate the saturation vapour pressure
    f = 1. - g_vpd * (es - ea)
    f = np.clip(f, 0, 1)  # Ensure that the reduction factor lies between 0 and 1
    return f


def temp_factor_Noilhan(T_A_K):
    """ Estimate stomatal stress due to temperature based on [Noilhan]_

    Parameteers
    -----------
    T_A_K : float
        Air temperature (Kelvin).

    Returns
    -------
    f : float
        Reduction factor in stomatal conductance [0-1]

    References
    ----------
    .. [Noilhan1989] J. Noilhan, S. Planton, A simple parameterization of
        land surface processes for meteorological models,
        Monthly Weather Review , Volume 117, 1989,
        Pages 536-549,
        https://doi.org/10.1175/1520-0493(1989)117<0536:ASPOLS>2.0.CO;2.
    """

    f = 1. - 0.0016 * (298 - T_A_K)**2
    f = np.clip(f, 0, 1)  # Ensure that the reduction factor lies between 0 and 1
    return f


def calc_T(H, T_A, R, rho_a, Cp):
    """ Calculate skin temperature by inversion of the equation for sensible heat transport

    Parameters
    ----------
    H : float
        Sensible heat flux (W m-2)
    T_A : float
        Air temperature (K)
    R : float
        Aerodynamic resistance (s m-1)
    rho_a : float
        Density of air (kg m-3)
    Cp : float
        Heat capacity of air at constant pressure (J kg-1 K-1)
    """

    T = T_A + H * R / (rho_a * Cp)
    return T


def calc_effective_resistances_SW(R_A, R_x, R_S, R_c, R_ss, delta, psicr):
    """ Calculate effective resistances to water vapour transport and soil
    and canopy contributions in ET for [Shuttleworth1985]_ model.

    Parameters
    ----------
    R_A : float
        Aerodynamic resistance to heat transport (s m-1)
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1)
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1)
    R_c : float
        Canopy bulk stomatal conductance (s m-1)
    Rss : float
        Resistance to water vapour transport in the soil surface (s m-1)
    delta : float
        Slope of the saturation water vapour pressure (kPa K-1)
    psicr : float
        Psicrometric constant (mb K-1)

    Returns
    -------
    R_a_SW : float
        Aerodynamic effective resistance to water transport (s m-1)
    R_s_SW : float
        Soil aerodynamic effective resistance to water transport (s m-1)
    R_c_SW : float
        Bulk canopy aerodynamic effective resistance to water transport (s m-1)
    C_s : float
        Contribution to LE by the soil source
    C_c : float
        Contribution to LE by the canopy source
    """

    delta_psicr = delta + psicr

    R_a_SW = delta_psicr * R_A                              # Eq. 16 [Shuttleworth1988]_
    R_s_SW = delta_psicr * R_S + psicr * R_ss                 # Eq. 17 [Shuttleworth1988]_
    R_c_SW = delta_psicr * R_x + psicr * R_c                  # Eq. 18 [Shuttleworth1988]_
    C_c = 1. / (1. + R_c_SW * R_a_SW / (R_s_SW * (R_c_SW + R_a_SW)))   # Eq. 14 [Shuttleworth1988]_
    C_s = 1. / (1. + R_s_SW * R_a_SW / (R_c_SW * (R_s_SW + R_a_SW)))   # Eq. 15 [Shuttleworth1988]_

    return R_a_SW, R_s_SW, R_c_SW, C_s, C_c

def calc_Ln(T_A_K, ea, f_cd=1):
    ''' Estimates net longwave radiation for potential ET
    Parameters
    ----------
    T_A_K : float or array
        Air temperature (Kelvin).
    u : float or array
        Wind speed above the canopy (m s-1).
    ea : float or array
        Water vapour pressure above the canopy (mb).
    f_cd : float or array
        cloudiness factor
    Returns
    -------
    Ln : float or array
        Net longwave radiation (W m-2)
    '''

    Ln = TSEB.rad.sb * f_cd * (0.34 - 0.14 * np.sqrt(ea*0.1)) * T_A_K**4

    return Ln


def calc_cloudiness(Sdn, S_0):

    f_cd = 1.35 * Sdn/S_0 - 0.35
    f_cd = np.clip(f_cd, 0.05, 1.0)
    return f_cd


def wind_profile(u, z_u, z_0M, d, z):

    u_z = u * np.log((z - d)/z_0M) / np.log((z_u - d)/z_0M)

    return u_z
