# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:58:34 2017

@author: Hector Nieto (hector.nieto@ica.csic.es)
"""
from collections import deque
import time

from . import TSEB
import numpy as np

# kB coefficient
KB_1_DEFAULT = 2.3

ITERATIONS = 15

TALL_REFERENCE = 1
SHORT_REFERENCE = 0

LOWEST_TC_DIFF = 5.  # Lowest Canopy to Air temperature difference
LOWEST_TS_DIFF = 5.  # Lowest Soil to Air temperature difference
F_LOW_TS_TC = 254  # Low Soil and Canopy Temperature flag
F_LOW_TS = 253  # Low Soil Temperature flag
F_LOW_TC = 252  # Low Canopy Temperature flag
T_DIFF_THRES = 0.1
STABILITY_THRES = -0.01


def penman_monteith(T_A_K,
                    u,
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
                    calcG_params=None,
                    const_L=None,
                    Rst_min=400,
                    leaf_type=TSEB.res.AMPHISTOMATOUS,
                    f_cd=None,
                    kB=2.3,
                    verbose=True):
    '''Penman Monteith [Allen1998]_ energy combination model.
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
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    Rst_min : float
        Minimum (unstressed) single-leaf stomatal resistance (s m -1), Default = 400 s m-1
    leaf_type : int
        1: Hipostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)
    f_cd : float or None
        cloudiness factor, if a value is set it will comput net Lw based on [Allen1998]_
    kB : float
        kB-1 parameter to compute roughness lenght for heat transport, default=2.3

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

    '''

    # Convert float scalars into numpy arrays and check parameters size
    if calcG_params is None:
        calcG_params = [[1], 0.35]

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
     leaf_type] = map(TSEB._check_default_parameter_size,
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
                       calcG_params[1],
                       Rst_min,
                       leaf_type],
                      [T_A_K] * 14)

    # Create the output variables
    [flag, Ln, LE, H, G, R_A, iterations] = [np.zeros(T_A_K.shape, np.float32) + np.nan for i in
                                             range(7)]

    # Calculate the general parameters
    rho_a = TSEB.met.calc_rho(p, ea, T_A_K)  # Air density
    Cp = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(
        T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_ = TSEB.met.calc_lambda(T_A_K)  # latent heat of vaporization MJ kg-1
    psicr = TSEB.met.calc_psicr(Cp, p, lambda_)  # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(T_A_K)  # saturation water vapour pressure in mb
    z_0H = TSEB.res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport
    vpd = es - ea

    # Calculate bulk stomatal resistance
    R_c = bulk_stomatal_resistance(LAI, Rst_min, leaf_type=leaf_type)

    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.full(T_A_K.shape, np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.full(T_A_K.shape, const_L)
        max_iterations = 1  # No iteration
    u_friction = TSEB.MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(TSEB.U_FRICTION_MIN, u_friction))
    L_queue = deque([np.array(L)], 6)
    L_converged = np.asarray(np.zeros(T_A_K.shape)).astype(bool)
    L_diff_max = np.inf
    zol = np.zeros(T_A_K.shape)
    T_0_K = T_A_K.copy()
    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = ~L_converged
        if np.all(L_converged):
            if verbose:
                if L_converged.size == 0:
                    print("Finished iterations with no valid solution")
                else:
                    print("Finished interations with a max. L diff: " + str(L_diff_max))
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        if verbose:
            print("Iteration: %d, non-converged pixels: "
                  "%d, max L diff: %f, total time: %f, loop time: %f" %
                  (n_iterations, np.sum(i), L_diff_max, total_duration, loop_duration))

        iterations[i] = n_iterations
        flag[i] = 0

        T_0_old = np.zeros(T_0_K.shape)
        for nn_interations in range(max_iterations):
            if f_cd is None:
                Ln = emis * (L_dn - TSEB.met.calc_stephan_boltzmann(T_0_K))
            else:
                # As the original equation in FAO56 uses net outgoing radiation
                Ln = - calc_Ln(T_A_K, ea, f_cd=f_cd)

            Rn = np.asarray(Sn + Ln)
            # Compute Soil Heat Flux
            G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn, i)
            # Calculate aerodynamic resistances
            R_A[i] = TSEB.res.calc_R_A(z_T[i], u_friction[i], L[i], d_0[i], z_0H[i])

            # Apply Penman Monteith Combination equation
            LE[i] = le_penman_monteith(Rn[i], G[i], vpd[i], R_A[i], R_c[i],
                                       delta[i], rho_a[i], Cp[i], psicr[i])
            H[i] = Rn[i] - G[i] - LE[i]

            # Recomputue aerodynamic temperature
            T_0_K[i] = calc_T(H[i], T_A_K[i], R_A[i], rho_a[i], Cp[i])
            if np.all(np.abs(T_0_K - T_0_old) < T_DIFF_THRES):
                break
            else:
                T_0_old = T_0_K.copy()

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
            # Check stability
            zol[i] = z_0M[i] / L[i]
            stable = np.logical_and(i, zol > STABILITY_THRES)
            L[stable] = 1e36
            # Calculate again the friction velocity with the new stability
            # correctios
            u_friction[i] = TSEB.MO.calc_u_star(u[i], z_u[i], L[i], d_0[i], z_0M[i])
            u_friction = np.asarray(np.maximum(TSEB.U_FRICTION_MIN, u_friction))
            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            i, L_queue, L_converged, L_diff_max = TSEB.monin_obukhov_convergence(L,
                                                                            L_queue,
                                                                            L_converged,
                                                                            flag)


    flag, T_0_K, Ln, LE, H, G, R_A, u_friction, L, n_iterations = map(
        np.asarray, (flag, T_0_K, Ln, LE, H, G, R_A, u_friction, L, n_iterations))

    return flag, T_0_K, Ln, LE, H, G, R_A, u_friction, L, n_iterations


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
                         f_g=1,
                         w_C=1,
                         Rst_min=100,
                         R_ss=500,
                         resistance_form=None,
                         calcG_params=None,
                         const_L=None,
                         massman_profile=None,
                         leaf_type=TSEB.res.AMPHISTOMATOUS,
                         kB=0,
                         verbose=True):
    '''Shuttleworth and Wallace [Shuttleworth1995]_ dual source energy combination model.
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
        Minimum (unstressed) single-leaf stomatal resistance (s m -1),
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
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).

    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    leaf_type : int
        1: Hipostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)
    kB : float
        kB-1 parameter to compute roughness lenght for heat transport, default=0

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
    '''

    # Convert float scalars into numpy arrays and check parameters size
    if massman_profile is None:
        massman_profile = [0, []]
    if calcG_params is None:
        calcG_params = [[1], 0.35]
    if resistance_form is None:
        resistance_form = [0, {}]

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
     f_g,
     w_C,
     Rst_min,
     R_ss,
     calcG_array,
     leaf_type] = map(TSEB._check_default_parameter_size,
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
                       f_g,
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
     Rn, Rn_C, Rn_S, C_s, C_c, PM_C, PM_S, iterations] = [np.full(T_A_K.shape, np.nan, np.float32)
                                                          for i in range(20)]

    # Calculate the general parameters
    rho_a = TSEB.met.calc_rho(p, ea, T_A_K)  # Air density
    Cp = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(
        T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_ = TSEB.met.calc_lambda(T_A_K)  # latent heat of vaporization MJ kg-1
    psicr = TSEB.met.calc_psicr(Cp, p, lambda_)  # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(T_A_K)  # saturation water vapour pressure in mb
    rho_cp = rho_a * Cp
    vpd = es - ea
    del es, ea

    F = np.asarray(LAI / f_c)  # Real LAI
    omega0 = TSEB.CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)

    # Calculate bulk stomatal resistance
    R_c = bulk_stomatal_resistance(LAI * f_g, Rst_min, leaf_type=leaf_type)
    del leaf_type, Rst_min

    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.full(T_A_K.shape, np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.full(T_A_K.shape, const_L)
        max_iterations = 1  # No iteration
    u_friction = TSEB.MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(TSEB.U_FRICTION_MIN, u_friction))
    L_queue = deque([np.array(L)], 6)
    L_converged = np.asarray(np.zeros(T_A_K.shape)).astype(bool)
    L_diff_max = np.inf
    z_0H = TSEB.res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport
    zol = np.zeros(T_A_K.shape)
    # First assume that temperatures equals the Air Temperature
    T_C, T_S, T_0 = T_A_K.copy(), T_A_K.copy(), T_A_K.copy()

    _, _, _, taudl = TSEB.rad.calc_spectra_Cambpell(LAI,
                                                    np.zeros(emis_C.shape),
                                                    1.0 - emis_C,
                                                    np.zeros(emis_S.shape),
                                                    1.0 - emis_S,
                                                    x_lad=x_LAD,
                                                    lai_eff=None)
    emiss = taudl * emis_S + (1 - taudl) * emis_C
    Ln = emiss * (L_dn - TSEB.met.calc_stephan_boltzmann(T_0))
    Ln_C = (1. - taudl) * Ln
    Ln_S = taudl * Ln

    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = ~L_converged
        if np.all(L_converged):
            if verbose:
                if L_converged.size == 0:
                    print("Finished iterations with no valid solution")
                else:
                    print("Finished interations with a max. L diff: " + str(L_diff_max))
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        if verbose:
            print("Iteration: %d, non-converged pixels: "
                  "%d, max L diff: %f, total time: %f, loop time: %f" %
                  (n_iterations, np.sum(i), L_diff_max, total_duration, loop_duration))

        iterations[i] = n_iterations
        flag[i] = 0

        T_C_old = np.zeros(T_C.shape)
        T_S_old = np.zeros(T_S.shape)
        for nn_interations in range(max_iterations):
            # Calculate aerodynamic resistances
            R_A[i], R_x[i], R_S[i] = TSEB.calc_resistances(resistance_form,
                                       {"R_A": {"z_T": z_T[i],
                                                "u_friction":
                                                    u_friction[i],
                                                "L": L[i],
                                                "d_0": d_0[i],
                                                "z_0H": z_0H[i],
                                                },
                                        "R_x": {"u_friction":
                                                    u_friction[i],
                                                "h_C": h_C[i],
                                                "d_0": d_0[i],
                                                "z_0M": z_0M[i],
                                                "L": L[i],
                                                "LAI": LAI[i],
                                                "leaf_width":
                                                    leaf_width[i],
                                                "massman_profile": massman_profile,
                                                "z0_soil": z0_soil[i],
                                                "res_params":
                                                    {k:res_params[k][i] for k in
                                                               res_params.keys()}
                                                },
                                        "R_S": {"u_friction": u_friction[i],
                                                'u': u[i],
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
                                                "deltaT": T_S[i] - T_0[i],
                                                "massman_profile": massman_profile,
                                                'rho': rho_a[i],
                                                'c_p': Cp[i],
                                                'f_cover': f_c[i],
                                                'w_C': w_C[i],
                                                "res_params":
                                                    {k: res_params[k][i] for k in
                                                               res_params.keys()}}
                                        }
                                                           )

            _, _, _, C_s[i], C_c[i] = calc_effective_resistances_SW(R_A[i],
                                                                    R_x[i],
                                                                    R_S[i],
                                                                    R_c[i],
                                                                    R_ss[i],
                                                                    delta[i],
                                                                    psicr[i])

            # Compute net bulk longwave radiation and split between canopy and soil
            Ln[i] = emiss[i] * (L_dn[i] - TSEB.met.calc_stephan_boltzmann(T_0[i]))
            Ln_C[i] = (1. - taudl[i]) * Ln[i]
            Ln_S[i] = taudl[i] * Ln[i]
            Rn_C[i] = Sn_C[i] + Ln_C[i]
            Rn_S[i] = Sn_S[i] + Ln_S[i]
            Rn[i] = Rn_C[i] + Rn_S[i]
            # Compute Soil Heat Flux Ratio
            G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn_S, i)

            # Eq. 12 in [Shuttleworth1988]_
            PM_C[i] = (delta[i] * (Rn[i] - G[i]) + (
                        rho_cp[i] * vpd[i] - delta[i] * R_x[i] * (Rn_S[i] - G[i])) / (
                               R_A[i] + R_x[i])) / \
                      (delta[i] + psicr[i] * (1. + R_c[i] / (R_A[i] + R_x[i])))

            # Avoid arithmetic error with no LAI
            PM_C[np.isnan(PM_C)] = 0
            # Eq. 13 in [Shuttleworth1988]_
            PM_S[i] = (delta[i] * (Rn[i] - G[i]) + (
                        rho_cp[i] * vpd[i] - delta[i] * R_S[i] * Rn_C[i]) / (
                                   R_A[i] + R_S[i])) / \
                      (delta[i] + psicr[i] * (1. + R_ss[i] / (R_A[i] + R_S[i])))
            PM_S[np.isnan(PM_S)] = 0
            # Eq. 11 in [Shuttleworth1988]_
            LE[i] = C_c[i] * PM_C[i] + C_s[i] * PM_S[i]
            H[i] = Rn[i] - G[i] - LE[i]

            # Compute canopy and soil  fluxes
            # Vapor pressure deficit at canopy source height (mb) # Eq. 8 in [Shuttleworth1988]_
            vpd_0[i] = vpd[i] + (
                        delta[i] * (Rn[i] - G[i]) - (delta[i] + psicr[i]) * LE[i]) * \
                       R_A[i] / (rho_cp[i])
            # Eq. 9 in Shuttleworth & Wallace 1985
            LE_S[i] = (delta[i] * (Rn_S[i] - G[i]) + rho_cp[i] * vpd_0[i] / R_S[i]) / \
                      (delta[i] + psicr[i] * (1. + R_ss[i] / R_S[i]))
            LE_S[np.isnan(LE_S)] = 0
            H_S[i] = Rn_S[i] - G[i] - LE_S[i]
            # Eq. 10 in Shuttleworth & Wallace 1985
            LE_C[i] = (delta[i] * Rn_C[i] + rho_cp[i] * vpd_0[i] / R_x[i]) / \
                      (delta[i] + psicr[i] * (1. + R_c[i] / R_x[i]))
            H_C[i] = Rn_C[i] - LE_C[i]
            no_canopy = np.logical_and(i, np.isnan(LE_C))
            H_C[no_canopy] = np.nan
            T_0[i] = calc_T(H[i], T_A_K[i], R_A[i], rho_a[i], Cp[i])
            T_C[i] = calc_T(H_C[i], T_0[i], R_x[i], rho_a[i], Cp[i])
            T_S[i] = calc_T(H_S[i], T_0[i], R_S[i], rho_a[i], Cp[i])
            no_valid_T = np.logical_and.reduce((i, T_C <= T_A_K - LOWEST_TC_DIFF,
                                                T_S <= T_A_K - LOWEST_TS_DIFF))
            flag[no_valid_T] = F_LOW_TS_TC
            T_C[no_valid_T] = T_A_K[no_valid_T] - LOWEST_TC_DIFF
            T_S[no_valid_T] = T_A_K[no_valid_T] - LOWEST_TS_DIFF
            no_valid_T = np.logical_and(i, T_C <= T_A_K - LOWEST_TC_DIFF)
            flag[no_valid_T] = F_LOW_TC
            T_C[no_valid_T] = T_A_K[no_valid_T] - LOWEST_TC_DIFF
            no_valid_T = np.logical_and(i, T_S <= T_A_K - LOWEST_TS_DIFF)
            flag[no_valid_T] = F_LOW_TS
            T_S[no_valid_T] = T_A_K[no_valid_T] - LOWEST_TS_DIFF

            if np.all(np.abs(T_C - T_C_old) < T_DIFF_THRES) \
                    and np.all(np.abs(T_S - T_S_old) < T_DIFF_THRES):
                break
            else:
                T_C_old = T_C.copy()
                T_S_old = T_S.copy()

        # Now L can be recalculated and the difference between iterations
        # derived
        if const_L is None:
            L[i] = TSEB.MO.calc_mo_length(u_friction[i],
                                          T_A_K[i],
                                          rho_a[i],
                                          Cp[i],
                                          H[i])
            zol[i] = z_0M[i] / L[i]
            stable = np.logical_and(i, zol > STABILITY_THRES)
            L[stable] = 1e36

            # Calculate again the friction velocity with the new stability
            # correctios
            u_friction[i] = TSEB.MO.calc_u_star(u[i], z_u[i], L[i], d_0[i], z_0M[i])
            u_friction[i] = np.asarray(np.maximum(TSEB.U_FRICTION_MIN, u_friction[i]))

            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            i, L_queue, L_converged, L_diff_max = TSEB.monin_obukhov_convergence(
                L,
                L_queue,
                L_converged,
                flag)

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

    return flag, T_S, T_C, vpd_0, Ln_S, Ln_C, LE, H, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L, n_iterations


def penman(T_A_K,
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
           calcG_params=[[1], 0.35],
           const_L=None,
           f_cd=None,
           kB=0,
           verbose=True):
    '''Penman energy combination model.
    Calculates the Penman evaporation fluxes using meteorological data.

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
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    f_cd : float or None
        cloudiness factor, if a value is set it will comput net Lw based on [Allen1998]_
    kB : float
        kB-1 parameter to compute roughness lenght for heat transport, default=2.3

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
    .. [Monteith2008] Monteith, JL, Unsworth MH, Principles of Environmental
    Physics, 2008. ISBN 978-0-12-505103-5

    '''

    # Convert float scalars into numpy arrays and check parameters size
    T_A_K = np.asarray(T_A_K)
    [u,
     ea,
     p,
     Sn,
     L_dn,
     emis,
     z_0M,
     d_0,
     z_u,
     z_T,
     calcG_array] = map(TSEB._check_default_parameter_size,
                      [u,
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
                      [T_A_K] * 11)

    # Create the output variables
    [flag, Ln, LE, H, G, R_A, iterations] = [np.zeros(T_A_K.shape, np.float32) + np.nan for i in
                                             range(7)]

    # Calculate the general parameters
    rho_a = TSEB.met.calc_rho(p, ea, T_A_K)  # Air density
    Cp = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air
    # slope of saturation water vapour pressure in mb K-1
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(T_A_K)
    lambda_ = TSEB.met.calc_lambda(T_A_K)  # latent heat of vaporization MJ kg-1
    psicr = TSEB.met.calc_psicr(Cp, p, lambda_)  # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(T_A_K)  # saturation water vapour pressure in mb
    z_0H = TSEB.res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport
    r_r = TSEB.res.calc_r_r(p, ea, T_A_K)  # Resitance to radiative transfer
    vpd = es - ea

    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.full(T_A_K.shape, np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.full(T_A_K.shape, const_L)
        max_iterations = 1  # No iteration
    u_friction = TSEB.MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(TSEB.U_FRICTION_MIN, u_friction))
    L_queue = deque([np.array(L)], 6)
    L_converged = np.asarray(np.zeros(T_A_K.shape)).astype(bool)
    L_diff_max = np.inf
    zol = np.zeros(T_A_K.shape)
    T_0_K = T_A_K.copy()
    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = ~L_converged
        if np.all(L_converged):
            if verbose:
                if L_converged.size == 0:
                    print("Finished iterations with no valid solution")
                else:
                    print("Finished interations with a max. L diff: " + str(L_diff_max))
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        if verbose:
            print("Iteration: %d, non-converged pixels: "
                  "%d, max L diff: %f, total time: %f, loop time: %f" %
                  (n_iterations, np.sum(i), L_diff_max, total_duration, loop_duration))

        iterations[i] = n_iterations
        flag[i] = 0

        T_0_old = np.zeros(T_0_K.shape)
        for nn_interations in range(max_iterations):
            if f_cd is None:
                Ln = emis * (L_dn - TSEB.met.calc_stephan_boltzmann(T_0_K))
            else:
                # As the original equation in FAO56 uses net outgoing radiation
                Ln = - calc_Ln(T_A_K, ea, f_cd=f_cd)

            Rn = np.asarray(Sn + Ln)
            # Compute Soil Heat Flux
            G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn, i)
            # Calculate aerodynamic resistances
            R_A[i] = TSEB.res.calc_R_A(z_T[i], u_friction[i], L[i], d_0[i], z_0H[i])

            # Apply Penman Combination equation
            LE[i] = le_penman(Rn[i], G[i], vpd[i], R_A[i], r_r[i], delta[i],
                              rho_a[i], Cp[i], psicr[i])
            H[i] = Rn[i] - G[i] - LE[i]

            # Recomputue aerodynamic temperature
            T_0_K[i] = calc_T(H[i], T_A_K[i], R_A[i], rho_a[i], Cp[i])
            if np.all(np.abs(T_0_K - T_0_old) < T_DIFF_THRES):
                break
            else:
                T_0_old = T_0_K.copy()

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
            # Check stability
            zol[i] = z_0M[i] / L[i]
            stable = np.logical_and(i, zol > STABILITY_THRES)
            L[stable] = 1e36
            # Calculate again the friction velocity with the new stability
            # correctios
            u_friction[i] = TSEB.MO.calc_u_star(u[i], z_u[i], L[i], d_0[i], z_0M[i])
            u_friction = np.asarray(np.maximum(TSEB.U_FRICTION_MIN, u_friction))
            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            L_new = L.copy()
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

    flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations = map(
        np.asarray, (flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations))

    return flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations


def pet_asce(T_A_K, u, ea, p, Sdn, z_u, z_T, f_cd=1, reference=TALL_REFERENCE,
             is_daily=True):
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
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(
        T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_ = TSEB.met.calc_lambda(T_A_K)  # latent heat of vaporization MJ kg-1
    c_p = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air
    psicr = TSEB.met.calc_psicr(c_p, p, lambda_)  # Psicrometric constant (mb K-1)
    es = TSEB.met.calc_vapor_pressure(
        T_A_K)  # saturation water vapour pressure in mb

    # Net shortwave radiation
    # Sdn = Sdn * 3600 / 1e6 # W m-2 to MJ m-2 h-1
    albedo = 0.23
    Sn = Sdn * (1.0 - albedo)
    # Net longwave radiation
    Ln = - calc_Ln(T_A_K, ea, f_cd=f_cd)
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

    if is_daily:
        G_ratio = 0
    # Soil heat flux
    G = G_ratio * Rn
    # Windspeed at 2m height
    z_0M = h_c * 0.123
    d = h_c * 0.67
    u_2 = wind_profile(u, z_u, z_0M, d, 2.0)

    LE = (delta * (Rn - G) + psicr * C_n * u_2 * (es - ea) / T_A_K) / (
                delta + psicr * C_d * u_2)

    return LE


def pet_fao56(T_A_K,
              u,
              ea,
              es,
              p,
              Sdn,
              z_u,
              z_T,
              f_cd=1,
              is_daily=True):
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
    delta = 10. * TSEB.met.calc_delta_vapor_pressure(
        T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_ = TSEB.met.calc_lambda(T_A_K)  # latent heat of vaporization MJ kg-1
    c_p = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air
    psicr = TSEB.met.calc_psicr(c_p, p, lambda_)  # Psicrometric constant (mb K-1)
    rho = TSEB.met.calc_rho(p, ea, T_A_K)

    # Net shortwave radiation
    # Sdn = Sdn * 3600 / 1e6 # W m-2 to MJ m-2 h-1
    albedo = 0.23
    Sn = Sdn * (1.0 - albedo)
    # Net longwave radiation
    Ln = - calc_Ln(T_A_K, ea,
                   f_cd=f_cd)  # As the original equation in FAO56 uses net outgoing radiation

    # Net radiation
    Rn = Sn + Ln

    if is_daily is True:
        G_ratio = 0
    else:
        G_ratio = np.zeros(Sdn.shape)
        case = Sdn > 0
        G_ratio[case] = 0.1
        G_ratio[~case] = 0.5

    h_c = 0.12
    R_c = 70.0

    # Soil heat flux
    G = G_ratio * Rn
    z_0M = h_c * 0.123
    z_0H = 0.1 * z_0M
    d = h_c * 2. / 3.
    # Windspeed at 2m height
    # u_2 = wind_profile(u, z_u, z_0M, d, 2.0)
    # R_a = 208. / u_2
    u_friction = TSEB.MO.calc_u_star(u, z_u, np.inf, d, z_0M)
    R_a = np.log((z_T - d) / z_0H) / (u_friction * TSEB.res.KARMAN)

    LE = le_penman_monteith(Rn, G, es - ea, R_a, R_c, delta, rho, c_p, psicr)

    return LE


def le_penman_monteith(r_n, g_flux, vpd, r_a, r_c, delta, rho, cp, psicr):
    r_eff = r_c / r_a
    le = ((delta * (r_n - g_flux) + rho * cp * vpd / r_a)
          / (delta + psicr * (1. + r_eff)))
    return le


def le_penman(r_n, g_flux, vpd, r_a, r_r, delta, rho, cp, psicr):

    r_hr = r_a + r_r
    psicr_star = psicr * r_a / r_hr
    le = ((delta * (r_n - g_flux) + rho * cp * vpd / r_hr)
          / (delta + psicr_star))
    return le


def bulk_stomatal_resistance(LAI, Rst, leaf_type=TSEB.res.AMPHISTOMATOUS):
    ''' Calculate the bulk canopy stomatal resistance.

    Parameters
    ----------
    LAI : float
        Leaf Area Index (m2 m-2).
    Rst: float
        Minimum (unstressed) single-leaf stomatal resistance (s m -1)
    leaf_type : int
        1: Hipostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)


    Returns
    -------
    R_c : float
        Canopy bulk stomatal resistance (s m-1)
    '''

    R_c = Rst / (leaf_type * LAI)
    return np.asarray(R_c)


def calc_T(H, T_A, R, rho_a, Cp):
    ''' Calculate skin temperature by inversion of the equation for sensible heat transport

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

    '''

    T = T_A + H * R / (rho_a * Cp)
    return T


def calc_effective_resistances_SW(R_A, R_x, R_S, R_c, R_ss, delta, psicr):
    ''' Calculate effective resistances to water vapour transport and soil
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
        Canopy bulk stomatal resistance (s m-1)
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
    '''

    delta_psicr = delta + psicr

    R_a_SW = delta_psicr * R_A  # Eq. 16 [Shuttleworth1988]_
    R_s_SW = delta_psicr * R_S + psicr * R_ss  # Eq. 17 [Shuttleworth1988]_
    R_c_SW = delta_psicr * R_x + psicr * R_c  # Eq. 18 [Shuttleworth1988]_
    C_c = 1. / (1. + R_c_SW * R_a_SW / (
                R_s_SW * (R_c_SW + R_a_SW)))  # Eq. 14 [Shuttleworth1988]_
    C_s = 1. / (1. + R_s_SW * R_a_SW / (
                R_c_SW * (R_s_SW + R_a_SW)))  # Eq. 15 [Shuttleworth1988]_

    C_c[np.isnan(C_c)] = 0
    C_s[np.isnan(C_s)] = 0
    return R_a_SW, R_s_SW, R_c_SW, C_s, C_c


def calc_Ln(T_A_K, ea, f_cd=1):
    ''' Estimates net emitted longwave radiation for potential ET
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
        Net outgoing flux of longwave radiation (W m-2)
    '''

    Ln = TSEB.rad.SB * f_cd * (0.34 - 0.14 * np.sqrt(ea * 0.1)) * T_A_K ** 4

    return Ln


def solar_radiation_clear_sky(doy, lat, elev, SOLAR_CONSTANT=1367):
    lat = np.radians(lat)
    doy_rad = 2. * np.pi * doy / 365.
    d_r = 1 + 0.033 * np.cos(doy_rad)
    decl = 0.409 * np.sin(doy_rad - 1.39)
    w_s = np.arccos(-np.tan(lat) * np.tan(decl))
    rad_0 = (SOLAR_CONSTANT * d_r / np.pi) * (w_s * np.sin(lat) * np.sin(decl) +
                                              np.cos(lat) * np.cos(decl) * np.sin(
                w_s))
    sdn_0 = (0.75 + 2e-5 * elev) * rad_0
    return sdn_0


def calc_cloudiness(sdn, lat, elev, doy):
    sdn_0 = solar_radiation_clear_sky(doy, lat, elev, SOLAR_CONSTANT=1367)
    f_cd = 1.35 * sdn / sdn_0 - 0.35
    f_cd = np.clip(f_cd, 0.05, 1.0)
    return f_cd


def wind_profile(u, z_u, z_0M, d, z):
    u_z = u * np.log((z - d) / z_0M) / np.log((z_u - d) / z_0M)

    return u_z


def fill_and_update_et(k_cs, et, et_ref, gaps):
    """
    Fills gaps on daily ET based on previous crop stress coefficient
    .. math : ET_{a,1} = k_{cs, 0} ET_{ref, 1}
    .. math : k_{cs, 1} = ET_{a,1} / ET_{ref, 1}
    Parameters
    ----------
    k_cs : ndarray
        Prior crop stress coefficient
        .. math : k_{cs, 0} = ET_{a,0} / ET_{ref,0}
    et : ndarray
        Daily actual ET
    et_ref : ndarray
        Daily reference ET
    gaps : ndarray bool
        Boolean array with True where gaps are present
    Returns
    -------
    et_filled : ndarray
        Daily ET with gaps filled
    kcs_updated : ndarray
        Updated crop stress coefficient based on valid `et` observations
    """

    # Only apply the method on valid reference ET observations, e.g. exclude oceans
    ref_valid = np.logical_and(np.isfinite(et_ref), et_ref >= 0)
    valid = np.logical_and(~gaps, ref_valid)
    no_valid = np.logical_and(gaps, ref_valid)
    # Fill gaps in daily ET
    et_filled = et.copy()
    del et
    et_filled[no_valid] = k_cs[no_valid] * et_ref[no_valid]
    kcs_updated = k_cs.copy()
    del k_cs
    # Update the K_cs coefficient with valid observations
    kcs_updated[valid] = et_filled[valid] / et_ref[valid]
    return et_filled, kcs_updated
