import numpy as np
from pyTSEB.TSEB import _check_default_parameter_size
from pyTSEB import meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import net_radiation as rad

# Convert from microEinstein to Watt
MUEINSTEIN_2_WATT = 0.219
# universal gas constant
GAS_CONSTANT = 8.31  # (J/K mol)
# intercellular O2 mol fraction (mmol mol-1)
OI = 210.
# Conductance ratios between CO2 and H2O
GV_GC_RATIO = 1.6
GV_GC_RATIO_CONV = 1.37
FILL_DATA = 9999.
N_ELEMENTS_CI = 1000
CO2_MOLAR_WEIGHT = 44
C_MOLAR_WEIGHT = 12.011

DEFAULT_C_KC = (404.9, 79430.0)  # Bonan2011
DEFAULT_C_KO = (278.4, 36380.0)  # Bonan2011
DEFAULT_C_TES = (42.75, 37830.0)  # Bonan2011
DEFAULT_C_VCX = (84.2, 65330.0, 149250.0, 485.0)  # Bonan2011
DEFAULT_C_RD = (0.015 * DEFAULT_C_VCX[0], 46390.0, 150650.0, 490.0)  # Bonan2011
DEFAULT_C_JX = (1.97 * DEFAULT_C_VCX[0], 43540.0, 152040.0, 495.0)  # Bonan2011
DEFAULT_C_TPU = (21.46, 53.100e3, 150650.0, 490.0)  # Bonan2011
A_GS = 11.0  # Wang1998
D_0_GS = 10.0  # hPa-1, Leuning 1995
G0P = 0.01  # mol H20 m-2 s-1
REL_DIFF = 0.001
N_ITERATIONS = 100
CANOPY_LAYERS = 10


def gpp_leaf_no_gs(t_a_k,
                   vpd,
                   apar,
                   ca=412.,
                   theta=0.9,
                   alpha=0.2,
                   c_kc=None,
                   c_ko=None,
                   c_tes=None,
                   c_rd=None,
                   c_vcx=None,
                   c_jx=None,
                   g0p=G0P,
                   a_1=A_GS,
                   d_0=D_0_GS,
                   fw=1,
                   verbose=True):
    """Model of photosynthesis Dewar+Farquhar.
    The function evaluates assimilation and conductance limited by temperature
    (Bernacchi's or Arrhenius equation) and radiation.

    Parameters
    ----------
    t_a_k : float or array_like
        Air temperature (Kelvin)
    vpd : float or array_like
        Vapor pressure deficit (mb)
    apar : float or array_like
        Absorbed photosynthetical active radiation
    ca : float or array_like
        Air CO2 concentration (micromol mol-1)
    theta : float
        Shape parameter for hyperbola (no units).
        Default 0.9 from [DiazEspejo2006]_
    alpha : float
        Quantum yield (mol electrons / mol quanta).
        Default 0.2 from [DiazEspejo2006]_
    c_kc : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for CO2 (no units).
        Default (38.05, 79430 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_ko : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for O2 (no units).
        Default (20.30, 36380 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_tes : tuple
        Reference value at 25ªC and activation energy (J mol–1) for
        CO2 compensation point in the absence of Rd (micromol mol-1)
        Default (19.02, 37830 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_rd : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        respiration (micromol m-2 s-1).
        Default (17.91, 44790 J mol-1) after [DiazEspejo2006]_ for olives
    c_vcx : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum catalytic activity of Rubisco in the presence of saturating
        amounts of RuBP and CO2 (micromol m-2 s-1).
        Default (33.99, 73680 J mol-1) after [DiazEspejo2006]_ for olives
    c_jx : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum rate of electron transport at saturating irradiance (micromol m-2 s-1)
        Default (18.88, 35350 J mol-1) after [DiazEspejo2006]_ for olives
    g0p : float
        Conductance at night (mmol/m2 s)
    a_1 : float
        Conversion factor gs [Leuning1995]_ model
    d_0 : float
        Stomata sensitivity parameter to VPD (mb) [Leuning1995]_ model
    oi : float or array_like, optional
        intercellular O2 mol fraction, default 210 micromol mol-1

    Returns
    -------
    gs : float or array_like
        C02 Stomatal conductance (mol/m2 s)
    assim : float or array_like
        Net Assimilation rate
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)

    References
    ----------
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    .. [Bernacchi2001] Bernacchi, C.J., Singsaas, E.L., Pimentel, C., Portis Jr,
        A.R. and Long, S.P., 2001.
        Improved temperature response functions for models of Rubisco‐limited
        photosynthesis.
        Plant, Cell & Environment, 24(2), pp.253-259.
        https://doi.org/10.1111/j.1365-3040.2001.00668.x
    .. [DiazEspejo2006] Díaz-Espejo, A., Walcroft, A.S., Fernández, J.E.,
        Hafidi, B., Palomo, M.J. and Girón, I.F., 2006.
        Modeling photosynthesis in olive leaves under drought conditions.
        Tree Physiology, 26(11), pp.1445-1456.
        https://doi.org/10.1093/treephys/26.11.1445
    .. [Medlyn2002] Medlyn, B.E., Dreyer, E., Ellsworth, D., Forstreuter,
        M., Harley, P.C., Kirschbaum, M.U.F., Le Roux, X., Montpied, P.,
        Strassemeyer, J., Walcroft, A. and Wang, K., 2002.
        Temperature response of parameters of a biochemically based model of
        photosynthesis. II. A review of experimental data.
        Plant, Cell & Environment, 25(9), pp.1167-1179.
        https://doi.org/10.1046/j.1365-3040.2002.00891.
    """
    [vpd, apar, ca, g0p, a_1, d_0, fw] = map(_check_default_parameter_size,
                                             [vpd, apar, ca, g0p, a_1, d_0, fw],
                                             7 * [t_a_k])

    vc_max, j_max, rd, kc, ko, tes_star, tpu = get_photosynthesis_params(t_a_k,
                                                                         c_kc=c_kc,
                                                                         c_ko=c_ko,
                                                                         c_tes=c_tes,
                                                                         c_rd=c_rd,
                                                                         c_vcx=c_vcx,
                                                                         c_jx=c_jx)

    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # (Eq. A3 [Leuning_1995]_)
    j_i = electron_transport_rate(apar, j_max, alpha=alpha, theta=theta)

    assim, gst, ci, temp_limited = gs_solver_leaf(vpd,
                                                 ca,
                                                 vc_max,
                                                 kc,
                                                 ko,
                                                 j_i,
                                                 tes_star,
                                                 rd,
                                                 g0p,
                                                 a_1,
                                                 d_0,
                                                 oi=OI,
                                                 fw=fw,
                                                  verbose=verbose)

    return assim, rd, gst, ci, temp_limited


def gs_solver_leaf(vpd, ca, vc_max, kc, ko, j_i, tes_star, rd, g0p, a_1, d_0, oi=OI,
                   fw=1, verbose=True):
    """Function to compute net assimilation and photosynthesis at potential
    values,i.e. without taking into account the stomatal conductance reduction
    associated with the leaf water potential

    Parameters
    ----------
    vpd : float or array_like
        Vapor pressure deficit (mb)
    ca : float or array_like
        Air CO2 concentration (micromol mol-1)
    vc_max : float or array_like
        Maximum catalytic activity of Rubisco in the presence of saturating
        levels of RuP^ and CO2 (micromol m-2 s-1)
    kc : float or array_like
        Michaelis constant for CO2 (no units)
    ko : float or array_like
        Michaelis constant for O2 (no units)
    j_i : float or array_like
        Electron transport rate for a given absorbed photon irradiance (micromol m-2 s-1)
    tes_star : float or array_like
        CO2 compensation point in the absence of dark respiration (micromol mol-1)
    rd : float or array_like
        Mitocondrial respiration (micromol m-2 s-1)
    g0p : float
        Conductance at night (mmol m-2 s-1)
    a_1 : float
        Conversion factor Leuning gs model
    d_0 : float
        Sensitivity of the stomata to VPD (mb)(Leuning,1995)
    oi : float
        intercellular O2 mol fraction

    Returns
    -------
    gs : float or array_like
        CO2 Stomatal conductance (mmol C02 m-2 s-1)
    assim : float or array_like
        Net Assimilation rate (micromol m-2 s-1)
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)

    References
    ----------
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    """

    # Initiate with ci = ca
    ci = np.full_like(vpd, ca)
    # Compensation point
    tes = compensation_point(tes_star, vc_max, rd, kc, ko, oi=oi)
    # First guess of canopy assimilation and canopy cod
    assim, temp_limited = calc_assim(ci, vc_max, kc, ko, j_i, tes_star, rd, oi=oi)
    gst = gs_ball_berry_leuning(assim, ca, vpd, tes=tes, g0p=g0p, a_1=a_1, d_0=d_0, fw=fw)
    gst = np.maximum(gst, g0p)
    # Convert H2O conductance to CO2
    gst /= GV_GC_RATIO
    assim_old = assim.copy()
    gst_old = gst.copy()
    ci_old = ci.copy()
    i = np.ones(vpd.shape, dtype=bool)
    n_iterations = 0
    while np.any(i) and n_iterations <= N_ITERATIONS:
        assim[i], temp_limited[i] = calc_assim(ci[i], vc_max[i], kc[i], ko[i], j_i[i],
                                               tes_star[i], rd[i], oi=oi)
        gst[i] = gs_ball_berry_leuning(assim[i], ca[i], vpd[i], tes=tes[i],
                                       g0p=g0p[i], a_1=a_1[i], d_0=d_0[i], fw=fw[i])

        gst[i] = np.maximum(gst[i], g0p[i])
        # Convert H2O conductance to CO2
        gst[i] /= GV_GC_RATIO
        ci[i] = ca[i] - assim[i] / gst[i]
        a_diff = np.abs((assim - assim_old) / assim_old)
        gst_diff = np.abs((gst - gst_old) / gst_old)
        ci_diff = np.abs((ci - ci_old) / ci_old)
        i = np.logical_or.reduce((a_diff > REL_DIFF,
                                  gst_diff > REL_DIFF,
                                  ci_diff > REL_DIFF))
        assim_old = assim.copy()
        gst_old = gst.copy()
        ci_old = ci.copy()
        n_iterations += 1
        if verbose:
            print(f"Iteration: {n_iterations}, non-converged pixels: {np.sum(i)}, "
                  f"max A diff: {np.nanmax(a_diff):4.3f}, "
                  f"max Gs diff: {np.nanmax(gst_diff):4.3f},"
                  f"max Ci diff: {np.nanmax(ci_diff):4.3f}")

    return assim, gst, ci, temp_limited


def gpp_canopy_no_gs(vpd,
                     r_x,
                     r_a,
                     t_a_k,
                     lai,
                     par_dir,
                     par_dif,
                     sza,
                     lai_eff=None,
                     x_lad=1,
                     rho_leaf=0.05,
                     tau_leaf=0.05,
                     rho_soil=0.15,
                     press=1013.15,
                     ca=412.,
                     theta=0.9,
                     alpha=0.2,
                     c_kc=None,
                     c_ko=None,
                     c_tes=None,
                     c_rd=None,
                     c_vcx=None,
                     c_jx=None,
                     c_tpu=None,
                     oi=OI,
                     f_soil=0,
                     kn=0.11,
                     kd_star=1e-6,
                     g0p=G0P,
                     a_1=A_GS,
                     d_0=D_0_GS,
                     fw=1,
                     leaf_type=1,
                     verbose=True):
    """Model of photosynthesis Dewar+Farquhar.
    The function evaluates assimilation and conductance limited by temperature
    (Bernacchi's or Arrhenius equation) and radiation.

    Parameters
    ----------
    gs : float or array_like
        C02 Canopy stomatal conductance (mol m-2 s-1)
    r_x : float or array_like
        Boundary canopy resistance to heat transport (s m-1)
    r_a : float or array_like
       Aerodynamic resistance to heat transport (s m-1)
    t_c_k : float or array_like
        Leaf effective temperature (Kelvin)
    t_a_k : float or array_like
        Air temperature (Kelvin)
    lai : float or array_like
        Leaf Area Index
    par_dir : float or array_like
        PAR beam irradiance at the top of the canopy (micromol m-2 s-1)
    par_dif : float or array_like
        PAR diffuse irradiance at the top of the canopy (micromol m-2 s-1)
    sza : float or array_like
        Sun Zenith Angle (degrees)
    lai_eff : float or array_like, optional
        Effective Leaf Area Index for beam radiation for a clumped canopy.
        Set None (default) for a horizontally homogenous canopy.
    x_lad : float or array_like, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float
        Soil bihemispherical reflectance
    press : float or array_like, optional
        Surface atmospheric pressure (mb), default = 1013.15 mb
    ca : float or array_like, optional
        Air CO2 concentration (micromol mol-1), default = 412 micromol mol-1
    theta : float, optional
        Shape parameter for hyperbola (no units).
        Default 0.9 from [DiazEspejo2006]_
    alpha : float, optional
        Quantum yield (mol electrons / mol quanta).
        Default 0.2 from [DiazEspejo2006]_
    c_kc : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for CO2 (no units).
        Default (38.05, 79430 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_ko : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for O2 (no units).
        Default (20.30, 36380 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_tes : tuple, optional
        Reference value at 25ªC and activation energy (J mol–1) for
        CO2 compensation point in the absence of Rd (micromol mol-1)
        Default (19.02, 37830 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_rd : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        respiration (micromol m-2 s-1).
        Default (17.91, 44790 J mol-1) after [DiazEspejo2006]_ for olives
    c_vcx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum catalytic activity of Rubisco in the presence of saturating
        amounts of RuBP and CO2 (micromol m-2 s-1).
        Default (33.99, 73680 J mol-1) after [DiazEspejo2006]_ for olives
    c_jx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum rate of electron transport at saturating irradiance (micromol m-2 s-1)
        Default (18.88, 35350 J mol-1) after [DiazEspejo2006]_ for olives
    oi : float or array_like, optional
        intercellular O2 mol fraction, default 210 micromol mol-1
    kn : float or array_like, optional
        Nitrogen decay coefficient through the canopy.
        Set to None for assuming homogeneous N distribution (e.g. short canopies)
        default = 0.11 after [Bonan2011]_
    f_soil : float or array_like, optional
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).
    leaf_type : int or array_like
        1: Hypostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)

    Returns
    -------
    assim : float or array_like
        Net Assimilation rate (micromol C02 m-2 s-1)
    rd : float or array_like
        Leaf respiration (micromol C02 m-2 s-1)
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)
    temp_limited : bool or array_like
        Flag for cases with photosynthesis limited by temperature

    References
    ----------
    .. [Wang1998] Wang, Y.-P., Leuning, R., 1998.
        A two-leaf model for canopy conductance, photosynthesis and partitioning
        of available energy I:
        Agricultural and Forest Meteorology 91, 89–111.
        https://doi.org/10.1016/S0168-1923(98)00061-6
    .. [Tuzet2003] Tuzet, A., Perrier, A., Leuning, R., 2003.
        A coupled model of stomatal conductance, photosynthesis and transpiration.
        Plant, Cell & Environment 26, 1097–1116.
        https://doi.org/10.1046/j.1365-3040.2003.01035.x
    .. [Bonan2011] Bonan, G.B., Lawrence, P.J., Oleson, K.W., Levis, S., Jung, M.,
        Reichstein, M., Lawrence, D.M., Swenson, S.C., 2011.
        Improving canopy processes in the Community Land Model version 4 (CLM4)
        using global flux fields empirically inferred from FLUXNET data.
        J. Geophys. Res. 116, G02014.
        https://doi.org/10.1029/2010JG001593
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    .. [Bernacchi2001] Bernacchi, C.J., Singsaas, E.L., Pimentel, C., Portis Jr,
        A.R. and Long, S.P., 2001.
        Improved temperature response functions for models of Rubisco‐limited
        photosynthesis.
        Plant, Cell & Environment, 24(2), pp.253-259.
        https://doi.org/10.1111/j.1365-3040.2001.00668.x

    """

    if lai_eff is None:
        lai_eff = lai.copy()

    [t_a_k, press, ca, f_soil, g0p, a_1, d_0, fw] = map(_check_default_parameter_size,
                                                        [t_a_k, press, ca, f_soil, g0p, a_1, d_0, fw],
                                                        8 * [vpd])

    # parameters of Farquar are calculated using functions of temperature
    vc_max, j_max, rd, kc, ko, tes_star, tpu = get_photosynthesis_params(t_a_k,
                                                                         c_kc=c_kc,
                                                                         c_ko=c_ko,
                                                                         c_tes=c_tes,
                                                                         c_rd=c_rd,
                                                                         c_vcx=c_vcx,
                                                                         c_jx=c_jx,
                                                                         c_tpu=c_tpu)

    ############################################################################
    # Integrate Photosyntesis variables based on [Wang1998]_
    ############################################################################
    # Beam extinction coefficient
    kb = rad.calc_K_be_Campbell(sza, x_lad)
    # Sunlint and shaded leaves
    lai_sunlit = canopy_integral(lai_eff, kb)  # Eq. 3 in [Dai2004]_
    lai_sunlit[~np.isfinite(lai_sunlit)] = 0.
    lai_sunlit = np.maximum(lai_sunlit, 0.)
    lai_shaded = lai - lai_sunlit  # Eq. 3 in [Dai2004]_
    w_sunlit = lai_sunlit / lai  # Eq. 4 in [Dai2004]_
    w_shaded = 1. - w_sunlit  # Eq. 4 in [Dai2004]_

    # Scaling up the parameters from single leaf to the big leaves
    # Eq. A11 in [Tuzet2013]_
    if not kn:
        kn = 1e-6

    # Canopy integral for V_max and Rd
    canopy_decay = canopy_integral(lai, kn)
    canopy_decay_sunlit = canopy_integral(lai_eff, kn + kb)
    canopy_decay_sunlit[np.isnan(canopy_decay_sunlit)] = 0
    canopy_decay_shaded = canopy_decay - canopy_decay_sunlit
    canopy_decay_shaded[np.isnan(canopy_decay_shaded)] = 0
    vc_max_sunlit = vc_max * canopy_decay_sunlit
    vc_max_shaded = vc_max * canopy_decay_shaded
    rd_sunlit = rd * canopy_decay_sunlit
    rd_shaded = rd * canopy_decay_shaded
    # Canopy spectral coefficients
    albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai,
                                                         sza,
                                                         rho_leaf,
                                                         tau_leaf,
                                                         rho_soil,
                                                         x_lad=x_lad,
                                                         lai_eff=lai_eff)

    # Canopy integral for Jc
    canopy_decay_sunlit = canopy_integral(lai_eff, kd_star + kb)
    canopy_decay_sunlit[np.isnan(canopy_decay_sunlit)] = 0
    canopy_decay_shaded = canopy_integral(lai, kd_star) - canopy_decay_sunlit
    canopy_decay_shaded[np.isnan(canopy_decay_shaded)] = 0
    jc_max_sunlit = j_max * canopy_decay_sunlit
    jc_max_shaded = j_max * canopy_decay_shaded

    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # (Eq. A3 [Leuning_1995]_)
    apar_sunlit = (1.0 - taubt) * (1.0 - albb) * par_dir + \
                  w_sunlit * (1.0 - taudt) * (1.0 - albd) * par_dif
    j_sunlit = electron_transport_rate(apar_sunlit, jc_max_sunlit, alpha=alpha, theta=theta)
    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # (Eq. A3 [Leuning_1995]_)
    apar_shaded = w_shaded * (1.0 - taudt) * (1.0 - albd) * par_dif
    j_shaded = electron_transport_rate(apar_shaded, jc_max_shaded, alpha=alpha, theta=theta)

    # Convert heat resistances to CO2 transport
    r_x = (2. / leaf_type) * res.molm2s1_2_ms1(t_a_k, press) * r_x * GV_GC_RATIO_CONV
    r_a = res.molm2s1_2_ms1(t_a_k, press) * r_a
    # Canopy resistance to CO2 trasnport corresponds to
    # the canopy stomatal and boundary layer resistances in series
    assim_shaded, gs_shaded, ci_shaded, temp_limited_shaded = gs_solver_canopy(vpd,
                                                                               lai_shaded,
                                                                               r_x / w_shaded,
                                                                               r_a,
                                                                               ca,
                                                                               vc_max_shaded,
                                                                               kc,
                                                                               ko,
                                                                               j_shaded,
                                                                               tes_star,
                                                                               rd_shaded,
                                                                               f_soil,
                                                                               g0p,
                                                                               a_1,
                                                                               d_0,
                                                                               oi=oi,
                                                                               fw=fw,
                                                                               leaf_type=leaf_type,
                                                                               verbose=verbose)

    assim_sunlit, gs_sunlit, ci_sunlit, temp_limited_sunlit = gs_solver_canopy(vpd,
                                                                               lai_sunlit,
                                                                               r_x / w_sunlit,
                                                                               r_a,
                                                                               ca,
                                                                               vc_max_sunlit,
                                                                               kc,
                                                                               ko,
                                                                               j_sunlit,
                                                                               tes_star,
                                                                               rd_sunlit,
                                                                               f_soil,
                                                                               g0p,
                                                                               a_1,
                                                                               d_0,
                                                                               oi=oi,
                                                                               fw=fw,
                                                                               leaf_type=leaf_type,
                                                                               verbose=verbose)

    assim = assim_shaded + assim_sunlit
    rd = rd_shaded + rd_sunlit
    gs = w_shaded * gs_shaded + w_sunlit * gs_sunlit
    ci = w_shaded * ci_shaded + w_sunlit * ci_sunlit
    temp_limited = w_shaded * temp_limited_shaded + w_sunlit * temp_limited_sunlit

    return assim, rd, gs, ci, temp_limited


def gs_solver_canopy(vpd, lai, r_x, r_a, ca, vc_max, kc, ko, j_c, tes_star, rd, r_soil, g0p, a_1, d_0,
                     oi=OI, fw=1, leaf_type=1, verbose=True):
    """

    Parameters
    ----------
    vpd
    lai
    r_x
    r_a
    ca
    vc_max
    kc
    ko
    j_c
    tes_star
    rd
    r_soil
    g0p
    a_1
    d_0
    oi
    fw
    leaf_type : int or array_like
        1: Hypostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)
    verbose

    Returns
    -------

    """
    # Initiate with ci = ca
    g0_c = lai * g0p * leaf_type
    ci = np.full_like(vpd, ca)

    # Compensation point
    tes = compensation_point(tes_star, vc_max, rd, kc, ko, oi=oi)
    # First guess of canopy assimilation and canopy cod
    assim, temp_limited = calc_assim(ci, vc_max, kc, ko, j_c, tes_star, rd, oi=oi)
    gs = gs_ball_berry_leuning(assim, ca, vpd, tes=tes, g0p=g0_c, a_1=a_1, d_0=d_0, fw=fw)
    gs = np.maximum(gs, g0_c)
    # Convert H2O conductance to CO2
    gs /= GV_GC_RATIO
    assim_old = assim.copy()
    gs_old = gs.copy()
    ci_old = ci.copy()
    i = np.ones(vpd.shape, dtype=bool)
    n_iterations = 0
    while np.any(i) and n_iterations <= N_ITERATIONS:
        r_c = r_x[i] + 1. / gs[i]
        assim[i], ci[i], temp_limited[i] = assim_tuzet(r_c, r_a[i], ca[i], vc_max[i],
                                                       kc[i], ko[i], j_c[i], tes_star[i],
                                                       rd[i], r_soil[i], oi=oi)
        cv = calc_cv(ci[i], ca[i], r_c, r_a[i], r_soil[i])
        gs[i] = gs_ball_berry_leuning(assim[i], cv, vpd[i], tes=tes[i],
                                      g0p=g0_c[i], a_1=a_1[i], d_0=d_0[i], fw=fw[i])

        gs[i] = np.maximum(gs[i], g0_c[i])
        # Convert H2O conductance to CO2
        gs[i] /= GV_GC_RATIO
        a_diff = np.abs((assim - assim_old) / assim_old)
        gs_diff = np.abs((gs - gs_old) / gs_old)
        ci_diff = np.abs((ci - ci_old) / ci_old)
        i = np.logical_or.reduce((a_diff > REL_DIFF,
                                  gs_diff > REL_DIFF,
                                  ci_diff > REL_DIFF))
        assim_old = assim.copy()
        gs_old = gs.copy()
        ci_old = ci.copy()
        n_iterations += 1
        if verbose:
            print(f"Iteration: {n_iterations}, non-converged pixels: {np.sum(i)}, "
                  f"max A diff: {np.nanmax(a_diff):4.3f}, "
                  f"max Gs diff: {np.nanmax(gs_diff):4.3f}, "
                  f"max Ci diff: {np.nanmax(ci_diff):4.3f}")

    # Compute effective leaf stomata conductance
    gst = gs / (lai * leaf_type)
    return assim, gst, ci, temp_limited


def gpp_canopy(gs,
               r_x,
               r_a,
               t_c_k,
               t_a_k,
               lai,
               par_dir,
               par_dif,
               sza,
               lai_eff=None,
               x_lad=1,
               rho_leaf=0.05,
               tau_leaf=0.05,
               rho_soil=0.15,
               press=1013.15,
               ca=412.,
               theta=0.9,
               alpha=0.2,
               c_kc=None,
               c_ko=None,
               c_tes=None,
               c_rd=None,
               c_vcx=None,
               c_jx=None,
               c_tpu=None,
               oi=OI,
               f_soil=0,
               kn=0.11,
               kd_star=1e-6,
               leaf_type=1):
    """Model of photosynthesis Dewar+Farquhar.
    The function evaluates assimilation and conductance limited by temperature
    (Bernacchi's or Arrhenius equation) and radiation.

    Parameters
    ----------
    gs : float or array_like
        C02 Canopy stomatal conductance (mol m-2 s-1)
    r_x : float or array_like
        Boundary canopy resistance to heat transport (s m-1)
    r_a : float or array_like
       Aerodynamic resistance to heat transport (s m-1)
    t_c_k : float or array_like
        Leaf effective temperature (Kelvin)
    t_a_k : float or array_like
        Air temperature (Kelvin)
    lai : float or array_like
        Leaf Area Index
    par_dir : float or array_like
        PAR beam irradiance at the top of the canopy (micromol m-2 s-1)
    par_dif : float or array_like
        PAR diffuse irradiance at the top of the canopy (micromol m-2 s-1)
    sza : float or array_like
        Sun Zenith Angle (degrees)
    lai_eff : float or array_like, optional
        Effective Leaf Area Index for beam radiation for a clumped canopy.
        Set None (default) for a horizontally homogenous canopy.
    x_lad : float or array_like, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float
        Soil bihemispherical reflectance
    press : float or array_like, optional
        Surface atmospheric pressure (mb), default = 1013.15 mb
    ca : float or array_like, optional
        Air CO2 concentration (micromol mol-1), default = 412 micromol mol-1
    theta : float, optional
        Shape parameter for hyperbola (no units).
        Default 0.9 from [DiazEspejo2006]_
    alpha : float, optional
        Quantum yield (mol electrons / mol quanta).
        Default 0.2 from [DiazEspejo2006]_
    c_kc : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for CO2 (no units).
        Default (38.05, 79430 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_ko : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for O2 (no units).
        Default (20.30, 36380 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_tes : tuple, optional
        Reference value at 25ªC and activation energy (J mol–1) for
        CO2 compensation point in the absence of Rd (micromol mol-1)
        Default (19.02, 37830 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_rd : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        respiration (micromol m-2 s-1).
        Default (17.91, 44790 J mol-1) after [DiazEspejo2006]_ for olives
    c_vcx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum catalytic activity of Rubisco in the presence of saturating
        amounts of RuBP and CO2 (micromol m-2 s-1).
        Default (33.99, 73680 J mol-1) after [DiazEspejo2006]_ for olives
    c_jx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum rate of electron transport at saturating irradiance (micromol m-2 s-1)
        Default (18.88, 35350 J mol-1) after [DiazEspejo2006]_ for olives
    oi : float or array_like, optional
        intercellular O2 mol fraction, default 210 micromol mol-1
    kn : float or array_like, optional
        Nitrogen decay coefficient through the canopy.
        Set to None for assuming homogeneous N distribution (e.g. short canopies)
        default = 0.11 after [Bonan2011]_
    f_soil : float or array_like, optional
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).
    leaf_type : int or array_like
        1: Hypostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)

    Returns
    -------
    assim : float or array_like
        Net Assimilation rate (micromol C02 m-2 s-1)
    rd : float or array_like
        Leaf respiration (micromol C02 m-2 s-1)
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)
    temp_limited : bool or array_like
        Flag for cases with photosynthesis limited by temperature

    References
    ----------
    .. [Wang1998] Wang, Y.-P., Leuning, R., 1998.
        A two-leaf model for canopy conductance, photosynthesis and partitioning
        of available energy I:
        Agricultural and Forest Meteorology 91, 89–111.
        https://doi.org/10.1016/S0168-1923(98)00061-6
    .. [Tuzet2003] Tuzet, A., Perrier, A., Leuning, R., 2003.
        A coupled model of stomatal conductance, photosynthesis and transpiration.
        Plant, Cell & Environment 26, 1097–1116.
        https://doi.org/10.1046/j.1365-3040.2003.01035.x
    .. [Bonan2011] Bonan, G.B., Lawrence, P.J., Oleson, K.W., Levis, S., Jung, M.,
        Reichstein, M., Lawrence, D.M., Swenson, S.C., 2011.
        Improving canopy processes in the Community Land Model version 4 (CLM4)
        using global flux fields empirically inferred from FLUXNET data.
        J. Geophys. Res. 116, G02014.
        https://doi.org/10.1029/2010JG001593
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    .. [Bernacchi2001] Bernacchi, C.J., Singsaas, E.L., Pimentel, C., Portis Jr,
        A.R. and Long, S.P., 2001.
        Improved temperature response functions for models of Rubisco‐limited
        photosynthesis.
        Plant, Cell & Environment, 24(2), pp.253-259.
        https://doi.org/10.1111/j.1365-3040.2001.00668.x

    """

    if  lai_eff is None:
        lai_eff = lai.copy()

    [t_c_k, press, ca, f_soil] = map(_check_default_parameter_size,
                          [t_c_k, press, ca, f_soil],
                          4 * [gs])

    # parameters of Farquar are calculated using functions of temperature
    vc_max, j_max, rd, kc, ko, tes_star, tpu = get_photosynthesis_params(t_c_k,
                                                                    c_kc=c_kc,
                                                                    c_ko=c_ko,
                                                                    c_tes=c_tes,
                                                                    c_rd=c_rd,
                                                                    c_vcx=c_vcx,
                                                                    c_jx=c_jx,
                                                                    c_tpu=c_tpu)

    ############################################################################
    # Integrate Photosyntesis variables based on [Wang1998]_
    ############################################################################
    # Beam extinction coefficient
    kb = rad.calc_K_be_Campbell(sza, x_lad)
    # Sunlint and shaded leaves
    lai_sunlit = canopy_integral(lai_eff, kb)  # Eq. 3 in [Dai2004]_
    lai_sunlit[~np.isfinite(lai_sunlit)] = 0.
    lai_sunlit = np.maximum(lai_sunlit, 0.)
    lai_shaded = lai - lai_sunlit  # Eq. 3 in [Dai2004]_
    w_sunlit = lai_sunlit / lai   # Eq. 4 in [Dai2004]_
    w_shaded = 1. - w_sunlit  # Eq. 4 in [Dai2004]_

    # Scaling up the parameters from single leaf to the big leaves
    # Eq. A11 in [Tuzet2013]_
    if not kn:
        kn = 1e-6

    canopy_decay = canopy_integral(lai, kn)
    rd *= canopy_decay
    tpu *= canopy_decay

    # Canopy integral for V_max
    canopy_decay_sunlit = canopy_integral(lai_eff, kn + kb)
    canopy_decay_sunlit[np.isnan(canopy_decay_sunlit)] = 0
    vc_max_sunlit = vc_max * canopy_decay_sunlit
    canopy_decay_shaded = canopy_decay - canopy_decay_sunlit
    canopy_decay_shaded[np.isnan(canopy_decay_shaded)] = 0
    vc_max_shaded = vc_max * canopy_decay_shaded
    vc_max_c = vc_max_sunlit + vc_max_shaded

    # Canopy spectral coefficients
    albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai,
                                                         sza,
                                                         rho_leaf,
                                                         tau_leaf,
                                                         rho_soil,
                                                         x_lad=x_lad,
                                                         lai_eff=lai_eff)

    # Canopy integral for Jc
    canopy_decay_sunlit = canopy_integral(lai_eff, kd_star + kb)
    canopy_decay_sunlit[np.isnan(canopy_decay_sunlit)] = 0
    canopy_decay_shaded = canopy_integral(lai, kd_star) - canopy_decay_sunlit
    canopy_decay_shaded[np.isnan(canopy_decay_shaded)] = 0
    jc_max_sunlit = j_max * canopy_decay_sunlit
    jc_max_shaded = j_max * canopy_decay_shaded

    apar_sunlit = (1.0 - taubt) * (1.0 - albb) * par_dir + \
                  w_sunlit * (1.0 - taudt) * (1.0 - albd) * par_dif

    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # (Eq. A3 [Leuning_1995]_)
    j_sunlit = electron_transport_rate(apar_sunlit, jc_max_sunlit, alpha=alpha, theta=theta)

    apar_shaded = w_shaded * (1.0 - taudt) * (1.0 - albd) * par_dif
    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # (Eq. A3 [Leuning_1995]_)
    j_shaded = electron_transport_rate(apar_shaded, jc_max_shaded, alpha=alpha, theta=theta)
    j_c = j_sunlit + j_shaded

    # Convert heat resistances to CO2 transport
    r_x = (2. / leaf_type) * res.molm2s1_2_ms1(t_a_k, press) * r_x * GV_GC_RATIO_CONV
    r_a = res.molm2s1_2_ms1(t_a_k, press) * r_a
    # Canopy resistance to CO2 trasnport corresponds to
    # the canopy stomatal and boundary layer resistances in series
    r_c = r_x + 1. / gs
    assim, ci, temp_limited = assim_tuzet(r_c, r_a, ca, vc_max_c, kc, ko, j_c,
                                          tes_star, rd, f_soil, tpu=tpu, oi=oi)

    return assim, rd, ci, temp_limited


def gpp_canopy_multilayer(gs,
                          r_x,
                          r_a,
                          t_c_k,
                          t_a_k,
                          lai,
                          par_dir,
                          par_dif,
                          sza,
                          lai_eff=None,
                          x_lad=1,
                          rho_leaf=0.05,
                          tau_leaf=0.05,
                          rho_soil=0.15,
                          press=1013.15,
                          ca=412.,
                          theta=0.9,
                          alpha=0.2,
                          c_kc=None,
                          c_ko=None,
                          c_tes=None,
                          c_rd=None,
                          c_vcx=None,
                          c_jx=None,
                          c_tpu=None,
                          oi=OI,
                          f_soil=0,
                          kn=0.11,
                          kd_star=1e-6,
                          leaf_type=1):
    """Model of photosynthesis Dewar+Farquhar.
    The function evaluates assimilation and conductance limited by temperature
    (Bernacchi's or Arrhenius equation) and radiation.

    Parameters
    ----------
    gs : float or array_like
        C02 Canopy stomatal conductance (mol m-2 s-1)
    r_x : float or array_like
        Boundary canopy resistance to heat transport (s m-1)
    r_a : float or array_like
       Aerodynamic resistance to heat transport (s m-1)
    t_c_k : float or array_like
        Leaf effective temperature (Kelvin)
    t_a_k : float or array_like
        Air temperature (Kelvin)
    lai : float or array_like
        Leaf Area Index
    par_dir : float or array_like
        PAR beam irradiance at the top of the canopy (micromol m-2 s-1)
    par_dif : float or array_like
        PAR diffuse irradiance at the top of the canopy (micromol m-2 s-1)
    sza : float or array_like
        Sun Zenith Angle (degrees)
    lai_eff : float or array_like, optional
        Effective Leaf Area Index for beam radiation for a clumped canopy.
        Set None (default) for a horizontally homogenous canopy.
    x_lad : float or array_like, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float
        Soil bihemispherical reflectance
    press : float or array_like, optional
        Surface atmospheric pressure (mb), default = 1013.15 mb
    ca : float or array_like, optional
        Air CO2 concentration (micromol mol-1), default = 412 micromol mol-1
    theta : float, optional
        Shape parameter for hyperbola (no units).
        Default 0.9 from [DiazEspejo2006]_
    alpha : float, optional
        Quantum yield (mol electrons / mol quanta).
        Default 0.2 from [DiazEspejo2006]_
    c_kc : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for CO2 (no units).
        Default (38.05, 79430 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_ko : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for O2 (no units).
        Default (20.30, 36380 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_tes : tuple, optional
        Reference value at 25ªC and activation energy (J mol–1) for
        CO2 compensation point in the absence of Rd (micromol mol-1)
        Default (19.02, 37830 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_rd : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        respiration (micromol m-2 s-1).
        Default (17.91, 44790 J mol-1) after [DiazEspejo2006]_ for olives
    c_vcx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum catalytic activity of Rubisco in the presence of saturating
        amounts of RuBP and CO2 (micromol m-2 s-1).
        Default (33.99, 73680 J mol-1) after [DiazEspejo2006]_ for olives
    c_jx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum rate of electron transport at saturating irradiance (micromol m-2 s-1)
        Default (18.88, 35350 J mol-1) after [DiazEspejo2006]_ for olives
    oi : float or array_like, optional
        intercellular O2 mol fraction, default 210 micromol mol-1
    kn : float or array_like, optional
        Nitrogen decay coefficient through the canopy.
        Set to None for assuming homogeneous N distribution (e.g. short canopies)
        default = 0.11 after [Bonan2011]_
    f_soil : float or array_like, optional
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).
    leaf_type : int or array_like
        1: Hypostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)

    Returns
    -------
    assim : float or array_like
        Net Assimilation rate (micromol C02 m-2 s-1)
    rd : float or array_like
        Leaf respiration (micromol C02 m-2 s-1)
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)
    temp_limited : bool or array_like
        Flag for cases with photosynthesis limited by temperature

    References
    ----------
    .. [Dai2004] Dai, Y., Dickinson, R.E., Wang, Y.-P., 2004.
        A Two-Big-Leaf Model for Canopy Temperature, Photosynthesis, and
        Stomatal Conductance.
        J. Climate 17, 2281–2299.
        https://doi.org/10.1175/1520-0442(2004)017<2281:ATMFCT>2.0.CO;2
    .. [Wang1998] Wang, Y.-P., Leuning, R., 1998.
        A two-leaf model for canopy conductance, photosynthesis and partitioning
        of available energy I:
        Agricultural and Forest Meteorology 91, 89–111.
        https://doi.org/10.1016/S0168-1923(98)00061-6
    .. [Tuzet2003] Tuzet, A., Perrier, A., Leuning, R., 2003.
        A coupled model of stomatal conductance, photosynthesis and transpiration.
        Plant, Cell & Environment 26, 1097–1116.
        https://doi.org/10.1046/j.1365-3040.2003.01035.x
    .. [Bonan2011] Bonan, G.B., Lawrence, P.J., Oleson, K.W., Levis, S., Jung, M.,
        Reichstein, M., Lawrence, D.M., Swenson, S.C., 2011.
        Improving canopy processes in the Community Land Model version 4 (CLM4)
        using global flux fields empirically inferred from FLUXNET data.
        J. Geophys. Res. 116, G02014.
        https://doi.org/10.1029/2010JG001593
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    .. [Bernacchi2001] Bernacchi, C.J., Singsaas, E.L., Pimentel, C., Portis Jr,
        A.R. and Long, S.P., 2001.
        Improved temperature response functions for models of Rubisco‐limited
        photosynthesis.
        Plant, Cell & Environment, 24(2), pp.253-259.
        https://doi.org/10.1111/j.1365-3040.2001.00668.x

    """

    if  lai_eff is None:
        lai_eff = lai.copy()

    [t_c_k, press, ca, f_soil] = map(_check_default_parameter_size,
                          [t_c_k, press, ca, f_soil],
                          4 * [gs])

    # parameters of Farquar are calculated using functions of temperature
    vc_max, j_max, rd, kc, ko, tes_star, tpu = get_photosynthesis_params(t_c_k,
                                                                    c_kc=c_kc,
                                                                    c_ko=c_ko,
                                                                    c_tes=c_tes,
                                                                    c_rd=c_rd,
                                                                    c_vcx=c_vcx,
                                                                    c_jx=c_jx,
                                                                    c_tpu=c_tpu)

    ############################################################################
    # Integrate Photosyntesis variables based on [Wang1998]_
    ############################################################################
    # Beam extinction coefficient
    kb = rad.calc_K_be_Campbell(sza, x_lad)
    sigma_leaf = rho_leaf + tau_leaf  # Leaf scattering coefficient
    abs_leaf_par = 1. - sigma_leaf  # leaf absorbtance

    # Scaling up the parameters from single leaf to the big leaves
    # Eq. A11 in [Tuzet2013]_
    if kn:
        canopy_decay = canopy_integral(lai, kn)
    else:
        canopy_decay = lai.copy()

    rd *= canopy_decay
    vc_max *= canopy_decay
    tpu *= canopy_decay

    # Initialize the integrated canopy electron transport rate variable
    # Top of the canopy electron transport rate
    apar = abs_leaf_par * (kb * par_dir + par_dif)

    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # (Eq. A3 [Leuning_1995]_)
    j_0 = electron_transport_rate(apar, j_max, alpha=alpha, theta=theta)
    j_c = np.zeros(gs.shape)
    delta_lai = lai / CANOPY_LAYERS  # LAI steps through the canopy depth
    for lai_i in np.linspace(lai / CANOPY_LAYERS, lai, CANOPY_LAYERS):
        lai_eff_i = lai_eff * lai_i / lai
        # PAR transmittance at a given canopy depth
        _, _, taubt, taudt = rad.calc_spectra_Cambpell(lai_i,
                                                       sza,
                                                       rho_leaf,
                                                       tau_leaf,
                                                       rho_soil,
                                                       x_lad=x_lad,
                                                       lai_eff=lai_eff_i)

        # fraction of sunlit and shaded leaf area within the canopy
        w_sunlit = np.exp(-lai_eff_i * kb)  # Eq. C3 in [Wang1998]_
        w_sunlit[np.isnan(w_sunlit)] = 0
        w_shaded = 1. - w_sunlit  # Eq. C4 in [Wang1998]_

        # PAR absorbed by a sunlit leaf at a given canopy depth
        apar = abs_leaf_par * kb * np.exp(-kb * lai_eff_i) * par_dir + abs_leaf_par * taudt * par_dif
        apar = np.maximum(0, apar)

        # calculate electron flux by solving the non-rectangular hyperbolic function:
        # (Eq. A3 [Leuning_1995]_)
        j_max_sunlit = j_max * np.exp(-lai_i * kd_star)
        j_sunlit = electron_transport_rate(apar, j_max_sunlit, alpha=alpha, theta=theta)

        # PAR absorbed by a shaded leaf at a given canopy depth
        apar = taudt * abs_leaf_par * par_dif
        apar = np.maximum(0, apar)
        # Decrease J_max according to the canopy profile for shaded leaves
        j_max_shaded = j_max * np.exp(-lai_i * kd_star)
        # calculate electron flux by solving the non-rectangular hyperbolic function:
        # (Eq. A3 [Leuning_1995]_)
        j_shaded = electron_transport_rate(apar, j_max_shaded, alpha=alpha, theta=theta)
        j_1 = w_sunlit * j_sunlit + w_shaded * j_shaded
        # Get the midpoint between the top and bottom of the canopy slice for the numerical integral
        j_i = 0.5 * j_0 + 0.5 * j_1
        # Update the top-layer electron transport rate for the next iteration
        j_0 = j_1.copy()
        # Add the canopy slice electron flux to the canopy integral
        j_c += (j_i * delta_lai)  # Eq. D4 in [Wang1998]_

    # Convert heat resistances to CO2 transport
    r_x = (2. / leaf_type) * res.molm2s1_2_ms1(t_a_k, press) * r_x * GV_GC_RATIO_CONV
    r_a = res.molm2s1_2_ms1(t_a_k, press) * r_a
    # Canopy resistance to CO2 trasnport corresponds to
    # the canopy stomatal and boundary layer resistances in series
    r_c = r_x + 1. / gs
    assim, ci, temp_limited = assim_tuzet(r_c, r_a, ca, vc_max, kc, ko, j_c,
                                          tes_star, rd, f_soil, tpu=tpu, oi=oi)

    return assim, rd, ci, temp_limited


def calc_assim(ci, vc_max, kc, ko, j_i, tes_star, rd, oi=OI):
    bb_t = kc * (1. + oi / ko)
    bb_r = 8. * tes_star
    # Assimilation temp limited(mol/ m2/s)
    # RuBP regeneration-limited carboxylation rate
    w_temp = w_func(ci, tes_star, vc_max, bb_t, 1.)
    # Assimilation rad limited(mol/ m2/s)
    # RuBP regeneration-limited carboxylation rate
    w_rad = w_func(ci, tes_star, j_i, bb_r, 4.)
    assim = np.minimum(w_temp, w_rad) - rd
    temp_limited = w_temp == assim
    return assim, temp_limited


def gpp_canopy_2leaf(gs,
                     r_x,
                     r_a,
                     t_c_k,
                     t_a_k,
                     lai,
                     par_dir,
                     par_dif,
                     sza,
                     lai_eff=None,
                     x_lad=1,
                     rho_leaf=0.05,
                     tau_leaf=0.05,
                     rho_soil=0.15,
                     press=1013.15,
                     ca=412.,
                     theta=0.9,
                     alpha=0.2,
                     c_kc=None,
                     c_ko=None,
                     c_tes=None,
                     c_rd=None,
                     c_vcx=None,
                     c_jx=None,
                     c_tpu=None,
                     oi=OI,
                     f_soil=0,
                     kn=0.11,
                     kd_star=1e-6,
                     leaf_type=1):
    """Model of photosynthesis Dewar+Farquhar.
    The function evaluates assimilation and conductance limited by temperature
    (Bernacchi's or Arrhenius equation) and radiation.

    Parameters
    ----------
    gs : float or array_like
        C02 Canopy stomatal conductance (mol m-2 s-1)
    r_x : float or array_like
        Boundary canopy resistance to heat transport (s m-1)
    r_a : float or array_like
       Aerodynamic resistance to heat transport (s m-1)
    t_c_k : float or array_like
        Leaf effective temperature (Kelvin)
    t_a_k : float or array_like
        Air temperature (Kelvin)
    lai : float or array_like
        Leaf Area Index
    par_dir : float or array_like
        PAR beam irradiance at the top of the canopy (micromol m-2 s-1)
    par_dif : float or array_like
        PAR diffuse irradiance at the top of the canopy (micromol m-2 s-1)
    sza : float or array_like
        Sun Zenith Angle (degrees)
    lai_eff : float or array_like, optional
        Effective Leaf Area Index for beam radiation for a clumped canopy.
        Set None (default) for a horizontally homogenous canopy.
    x_lad : float or array_like, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float
        Soil bihemispherical reflectance
    press : float or array_like, optional
        Surface atmospheric pressure (mb), default = 1013.15 mb
    ca : float or array_like, optional
        Air CO2 concentration (micromol mol-1), default = 412 micromol mol-1
    theta : float, optional
        Shape parameter for hyperbola (no units).
        Default 0.9 from [DiazEspejo2006]_
    alpha : float, optional
        Quantum yield (mol electrons / mol quanta).
        Default 0.2 from [DiazEspejo2006]_
    c_kc : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for CO2 (no units).
        Default (38.05, 79430 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_ko : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for O2 (no units).
        Default (20.30, 36380 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_tes : tuple, optional
        Reference value at 25ªC and activation energy (J mol–1) for
        CO2 compensation point in the absence of Rd (micromol mol-1)
        Default (19.02, 37830 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_rd : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        respiration (micromol m-2 s-1).
        Default (17.91, 44790 J mol-1) after [DiazEspejo2006]_ for olives
    c_vcx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum catalytic activity of Rubisco in the presence of saturating
        amounts of RuBP and CO2 (micromol m-2 s-1).
        Default (33.99, 73680 J mol-1) after [DiazEspejo2006]_ for olives
    c_jx : tuple, optional
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum rate of electron transport at saturating irradiance (micromol m-2 s-1)
        Default (18.88, 35350 J mol-1) after [DiazEspejo2006]_ for olives
    oi : float or array_like, optional
        intercellular O2 mol fraction, default 210 micromol mol-1
    kn : float or array_like, optional
        Nitrogen decay coefficient through the canopy.
        default = 0.11 after [Bonan2011]_
    f_soil : float or array_like, optional
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).
    leaf_type : int or array_like
        1: Hypostomatous leaves (stomata only in one side of the leaf)
        2: Amphistomatous leaves (stomata in both sides of the leaf)

    Returns
    -------
    assim : float or array_like
        Net Assimilation rate (micromol C02 m-2 s-1)
    rd : float or array_like
        Leaf respiration (micromol C02 m-2 s-1)
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)
    temp_limited : bool or array_like
        Flag for cases with photosynthesis limited by temperature

    References
    ----------
    .. [Dai2004] Dai, Y., Dickinson, R.E., Wang, Y.-P., 2004.
        A Two-Big-Leaf Model for Canopy Temperature, Photosynthesis, and
        Stomatal Conductance.
        J. Climate 17, 2281–2299.
        https://doi.org/10.1175/1520-0442(2004)017<2281:ATMFCT>2.0.CO;2
    .. [Wang1998] Wang, Y.-P., Leuning, R., 1998.
        A two-leaf model for canopy conductance, photosynthesis and partitioning
        of available energy I:
        Agricultural and Forest Meteorology 91, 89–111.
        https://doi.org/10.1016/S0168-1923(98)00061-6
    .. [Bonan2011] Bonan, G.B., Lawrence, P.J., Oleson, K.W., Levis, S., Jung, M.,
        Reichstein, M., Lawrence, D.M., Swenson, S.C., 2011.
        Improving canopy processes in the Community Land Model version 4 (CLM4)
        using global flux fields empirically inferred from FLUXNET data.
        J. Geophys. Res. 116, G02014.
        https://doi.org/10.1029/2010JG001593
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    .. [Bernacchi2001] Bernacchi, C.J., Singsaas, E.L., Pimentel, C., Portis Jr,
        A.R. and Long, S.P., 2001.
        Improved temperature response functions for models of Rubisco‐limited
        photosynthesis.
        Plant, Cell & Environment, 24(2), pp.253-259.
        https://doi.org/10.1111/j.1365-3040.2001.00668.x
    """

    if  lai_eff is None:
        lai_eff = lai.copy()

    [t_c_k, press, ca, f_soil] = map(_check_default_parameter_size,
                          [t_c_k, press, ca, f_soil],
                          4 * [gs])

    # parameters of Farquar are calculated using functions of temperature
    vc_max, j_max, rd, kc, ko, tes_star, tpu = get_photosynthesis_params(t_c_k,
                                                                    c_kc=c_kc,
                                                                    c_ko=c_ko,
                                                                    c_tes=c_tes,
                                                                    c_rd=c_rd,
                                                                    c_vcx=c_vcx,
                                                                    c_jx=c_jx,
                                                                    c_tpu=c_tpu)

    ############################################################################
    # Integrate Photosyntesis variables based on [Dai2004]_
    ############################################################################
    # Beam extinction coefficient
    kb = rad.calc_K_be_Campbell(sza, x_lad)
    # Sunlint and shaded leaves
    lai_sunlit = canopy_integral(lai_eff, kb)  # Eq. 3 in [Dai2004]_
    lai_sunlit[np.isnan(lai_sunlit)] = 0.
    lai_sunlit = np.maximum(lai_sunlit, 0.)
    lai_shaded = lai - lai_sunlit  # Eq. 3 in [Dai2004]_
    w_sunlit = lai_sunlit / lai   # Eq. 4 in [Dai2004]_
    w_shaded = 1. - w_sunlit  # Eq. 4 in [Dai2004]_

    # Scaling up the parameters from single leaf to the big leaves
    # Canopy integral for Vc_max and Respiration
    if not kn:
        kn = 1e-6
    canopy_decay_sunlit = canopy_integral(lai_eff, kn + kb)  # Eq. C6 in [Wang1998]_
    canopy_decay_sunlit[np.isnan(canopy_decay_sunlit)] = 0
    canopy_decay_shaded = canopy_integral(lai, kn) - canopy_decay_sunlit  # Eq. C7 in [Wang1998]_
    canopy_decay_shaded[np.isnan(canopy_decay_shaded)] = 0
    rd_sunlit = rd * canopy_decay_sunlit
    rd_shaded = rd * canopy_decay_shaded
    vc_max_sunlit = vc_max * canopy_decay_sunlit
    vc_max_shaded = vc_max * canopy_decay_shaded
    tpu_sunlit = tpu * canopy_decay_sunlit
    tpu_shaded = tpu * canopy_decay_shaded

    # Canopy spectral coefficients
    albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai,
                                                         sza,
                                                         rho_leaf,
                                                         tau_leaf,
                                                         rho_soil,
                                                         x_lad=x_lad,
                                                         lai_eff=lai_eff)
    # Canopy integral for Jc
    canopy_decay_sunlit = canopy_integral(lai_eff, kd_star + kb)
    canopy_decay_sunlit[np.isnan(canopy_decay_sunlit)] = 0
    canopy_decay_shaded = canopy_integral(lai, kd_star) - canopy_decay_sunlit
    canopy_decay_shaded[np.isnan(canopy_decay_shaded)] = 0
    jc_max_sunlit = j_max * canopy_decay_sunlit
    jc_max_shaded = j_max * canopy_decay_shaded

    apar_sunlit = (1.0 - taubt) * (1.0 - albb) * par_dir + \
                  w_sunlit * (1.0 - taudt) * (1.0 - albd) * par_dif

    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # $\theta J^2  - ( \alpha Q + J_{lmax} ) J + \alpha Q J_{lmax} = 0$
    # (Eq. A3 [Leuning_1995]_)
    j_c_sunlit = electron_transport_rate(apar_sunlit, jc_max_sunlit, alpha=alpha, theta=theta)

    apar_shaded = w_shaded * (1.0 - taudt) * (1.0 - albd) * par_dif
    # calculate electron flux by solving the non-rectangular hyperbolic function:
    # (Eq. A3 [Leuning_1995]_)
    j_c_shaded = electron_transport_rate(apar_shaded, jc_max_shaded, alpha=alpha, theta=theta)

    # Convert heat resistances to CO2 transport
    r_x = (2. / leaf_type) * res.molm2s1_2_ms1(t_a_k, press) * r_x * GV_GC_RATIO_CONV
    r_a = res.molm2s1_2_ms1(t_a_k, press) * r_a
    # Canopy resistance to CO2 trasnport corresponds to
    # the canopy stomatal and boundary layer resistances in series
    r_c = (r_x + 1. / gs)
    assim_sunlit, ci_sunlit, temp_limited_sunlit = assim_tuzet(r_c / w_sunlit,
                                                               r_a,
                                                               ca,
                                                               vc_max_sunlit,
                                                               kc,
                                                               ko,
                                                               j_c_sunlit,
                                                               tes_star,
                                                               rd_sunlit,
                                                               f_soil,
                                                               tpu=tpu,
                                                               oi=oi)

    assim_shaded, ci_shaded, temp_limited_shaded = assim_tuzet(r_c / w_shaded,
                                                               r_a,
                                                               ca,
                                                               vc_max_shaded,
                                                               kc,
                                                               ko,
                                                               j_c_shaded,
                                                               tes_star,
                                                               rd_shaded,
                                                               f_soil,
                                                               tpu=tpu,
                                                               oi=oi)

    assim = assim_shaded + assim_sunlit
    rd = rd_shaded + rd_sunlit
    ci = w_shaded * ci_shaded + w_sunlit * ci_sunlit
    temp_limited = w_shaded * temp_limited_shaded \
                   + w_sunlit * temp_limited_sunlit

    return assim, rd, ci, temp_limited


def canopy_integral(lai, k):
    """

    Parameters
    ----------
    lai : float or array
        Leaf Area Index
    k : float or array
        Extinction coefficient

    Returns
    -------
    integral : float or array
        Integral function of extinction through a canopy layer
    References
    ----------
    .. [Wang1998] Wang, Y.-P., Leuning, R., 1998.
        A two-leaf model for canopy conductance, photosynthesis and partitioning
        of available energy I:
        Agricultural and Forest Meteorology 91, 89–111.
        https://doi.org/10.1016/S0168-1923(98)00061-6
    """
    # Eq. B5 in [Wang1998]_
    integral = (1. - np.exp(-lai * k)) / k
    return integral


def assim_tuzet(r_c, r_a, ca, vc_max, kc, ko, j_i, tes_star, rd, f_soil=0, tpu=np.nan, oi=OI):
    """Function to compute net assimilation and photosynthesis given canopy resistances
     without taking into account the stomatal conductance reduction
    associated with the leaf water potential

    Parameters
    ----------
    ca : float or array_like
        Air CO2 concentration (micromol mol-1)
    vc_max : float or array_like
        Maximum catalytic activity of Rubisco in the presence of saturating
        levels of RuP^ and CO2 (micromol m-2 s-1)
    kc : float or array_like
        Michaelis constant for CO2 (no units)
    ko : float or array_like
        Michaelis constant for O2 (no units)
    j_i : float or array_like
        Electron transport rate for a given absorbed photon irradiance (micromol m-2 s-1)
    tes_star : float or array_like
        CO2 compensation point in absence of dark respiration (micromol mol-1)
    rd : float or array_like
        Mitocondrial respiration (micromol m-2 s-1)
    f_soil : float or array_like, optional
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).
    oi : float or array_like, optional
        intercellular O2 mol fraction, default 210 micromol mol-1

    Returns
    -------
    assim : float or array_like
        Net Assimilation rate (micromol m-2 s-1)
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)
    temp_limited : bool or array_like
        Flag for cases with photosynthesis limited by temperature

    References
    ----------
    .. [Tuzet2003] Tuzet, A., Perrier, A., Leuning, R., 2003.
        A coupled model of stomatal conductance, photosynthesis and transpiration.
        Plant, Cell & Environment 26, 1097–1116.
        https://doi.org/10.1046/j.1365-3040.2003.01035.x
    """

    cv_term_1 = r_c * (ca + r_a * f_soil) / (r_a + r_c)
    cv_term_2 = r_a / (r_a + r_c)
    rd_scaled = r_c * rd
    # parameters for temp limitation (Eq. A1 in [Leuning_1995]_)
    vc_scaled = r_c * vc_max
    k_prime = kc * (1. + oi / ko)
    # Ci solution for temperature limitation
    a_t = cv_term_2 - 1.
    b_t = cv_term_1 + k_prime * (cv_term_2 - 1.) + rd_scaled - vc_scaled
    c_t = k_prime * (cv_term_1 + rd_scaled) + vc_scaled * tes_star
    ci_t_pos = (-b_t - np.sqrt(b_t**2 - 4 * a_t * c_t)) / (2. * a_t)

    # parameters for radiation limitation (Eq. A2 in [Leuning_1995]_)
    j_scaled = r_c * j_i / 4.
    # Ci solution for temperature limitation
    a_r = cv_term_2 - 1.
    b_r = cv_term_1 + 2. * tes_star * (cv_term_2 - 1.) + rd_scaled - j_scaled
    c_r = 2. * tes_star * (cv_term_1 + rd_scaled) + j_scaled * tes_star
    ci_r_pos = (-b_r - np.sqrt(b_r ** 2 - 4. * a_r * c_r)) / (2. * a_r)

    ci_sol = np.maximum(ci_r_pos, ci_t_pos)
    temp_limited = ci_sol == ci_t_pos
    # ci_sol = ci_t_pos
    cv_sol = calc_cv(ci_sol, ca, r_c, r_a, f_soil)
    assim = (cv_sol - ci_sol) / r_c
    return assim, ci_sol, temp_limited


def calc_cv(ci, ca, r_alpha, ra, f_soil):
    """ Computes the CO2 concentration at the canopy-air interface,
    assuming CO2 flux interaction between the canopy, air and the soil/background.

    Parameters
    ----------
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)
    ca : float or array_like
        Atmospheric C02 concentration above the canopy (micromol mol-1)
    r_alpha : float or array_like
        Canopy resistance (stomatal+boundary) to CO2 transport (m2 s1 mol-1)
    ra : float or array_like
        Aerodynamic resistance to CO2 transport  (m2 s1 mol-1)
    f_soil : float or array_like
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).
    Returns
    -------
    cv : float or array_like
        C02 concentration at the air-canopy interface (micromol mol-1)
    """
    cv = (ca / ra + ci / r_alpha + f_soil) / (1. / ra + 1. / r_alpha)
    return cv


def t_func(temp, cpar, delh):
    """Calculation of the parameters of Farquar equation under temperature
        limitation, based on [Bernacchi2001]_ equation.

    This equation assumes that, regardless of the amount of enzyme present or
    the activation state of the enzyme, the activity will continue to
    increase exponentially as temperature increases. [Bernacchi2001]_

    Parameters
    ----------
    temp : float or array_like
        Air temperature (K)
    cpar : float
        Normalized reference value at 25º C (no units)
    delh : float
        Activation energy (J mol – 1)

    Returns
    -------
    y : float or array_like
        Parameter adjusted to temperature response

    References
    ----------
    .. [Bernacchi2001] Bernacchi, C.J., Singsaas, E.L., Pimentel, C., Portis Jr,
        A.R. and Long, S.P., 2001.
        Improved temperature response functions for models of Rubisco‐limited
        photosynthesis.
        Plant, Cell & Environment, 24(2), pp.253-259.
        https://doi.org/10.1111/j.1365-3040.2001.00668.x

    """
    y = np.exp(cpar - delh / (GAS_CONSTANT * temp))  # Eq. 8 in [Bernacchi2001]_
    return y


def t_func_bernacchi03(temp, k_25, delh):
    """Calculation of the parameters of Farquar equation under temperature
        limitation, based on [Bernacchi2001]_ equation.

    This equation assumes that, regardless of the amount of enzyme present or
    the activation state of the enzyme, the activity will continue to
    increase exponentially as temperature increases. [Bernacchi2001]_

    Parameters
    ----------
    temp : float or array_like
        Air temperature (K)
    cpar : float
        Reference value at 25º C (no units)
    delh : float
        Activation energy (J mol – 1)

    Returns
    -------
    y : float or array_like
        Parameter adjusted to temperature response

    References
    ----------
    .. [Bernacchi2001] Bernacchi, C.J., Singsaas, E.L., Pimentel, C., Portis Jr,
        A.R. and Long, S.P., 2001.
        Improved temperature response functions for models of Rubisco‐limited
        photosynthesis.
        Plant, Cell & Environment, 24(2), pp.253-259.
        https://doi.org/10.1111/j.1365-3040.2001.00668.x

    """
    y = k_25 * np.exp(delh * (temp - 298.) / (GAS_CONSTANT * 298. * temp))  # Eq. 8 in [Bernacchi2001]_
    return y


def t_func_arrhenius(temp, k_25, delta_h, delta_hd=None, delta_s=490.):
    """Calculation of the parameters of Farquar equation under temperature
        limitation, based on [Medlyn2002]_ equation.

    Parameters
    ----------
    temp : float or array_like
        Air temperature (K)
    k_25 : float
        Reference value at 25º C (micromol m−2 s−1)
    delta_h : float
        Activation energy (J mol–1)
    delta_hd : float
        Deactivation energy (J mol-1).
        Default 200 000 J mol-1 after [Prieto2012]_
    delta_s : float
        Entropy term (J mol-1).
        Default 230 after [Bonan2011]_

    Returns
    -------
    y : float or array_like
        Parameter adjusted to temperature response (μmol m−2 s−1)

    References
    ----------
    .. [Medlyn2002] Medlyn, B.E., Dreyer, E., Ellsworth, D., Forstreuter, M.,
        Harley, P.C., Kirschbaum, M.U.F., Le Roux, X., Montpied, P.,
        Strassemeyer, J., Walcroft, A. and Wang, K., 2002.
        Temperature response of parameters of a biochemically based model of
        photosynthesis. II. A review of experimental data.
        Plant, Cell & Environment, 25(9), pp.1167-1179.
        https://doi.org/10.1046/j.1365-3040.2002.00891.x
    """
    f_t = temperature_function(temp, delta_h, t_ref=298.15)
    if delta_hd is None:
        f_d = 1
    else:
        f_d = temperature_inhibition(temp, delta_hd, delta_s, t_ref=298.15)

    y = k_25 * f_t * f_d
    return y


def temperature_function(temp, delta_h, t_ref=298.15):
    f_t = np.exp(delta_h * (1 - t_ref / temp) / (t_ref * GAS_CONSTANT))
    return f_t


def temperature_inhibition(temp, delta_hd, delta_s, t_ref=298.15):
    f_d = (1. + np.exp((t_ref * delta_s - delta_hd) / (t_ref * GAS_CONSTANT))) \
          / (1. + np.exp((temp * delta_s - delta_hd) / (temp * GAS_CONSTANT)))

    return f_d


def gs_solver(vpd, ca, vc_max, kc, ko, j_i, tes_star, rd, g0p, a_1, d_0, oi=OI):
    """Function to compute net assimilation and photosynthesis at potential
    values,i.e. without taking into account the stomatal conductance reduction
    associated with the leaf water potential

    Parameters
    ----------
    vpd : float or array_like
        Vapor pressure deficit (mb)
    ca : float or array_like
        Air CO2 concentration (micromol mol-1)
    vc_max : float or array_like
        Maximum catalytic activity of Rubisco in the presence of saturating
        levels of RuP^ and CO2 (micromol m-2 s-1)
    kc : float or array_like
        Michaelis constant for CO2 (no units)
    ko : float or array_like
        Michaelis constant for O2 (no units)
    j_i : float or array_like
        Electron transport rate for a given absorbed photon irradiance (micromol m-2 s-1)
    tes_star : float or array_like
        CO2 compensation point in the absence of dark respiration (micromol mol-1)
    rd : float or array_like
        Mitocondrial respiration (micromol m-2 s-1)
    g0p : float
        Conductance at night (mmol m-2 s-1)
    a_1 : float
        Conversion factor Leuning gs model
    d_0 : float
        Sensitivity of the stomata to VPD (mb)(Leuning,1995)
    oi : float
        intercellular O2 mol fraction

    Returns
    -------
    gs : float or array_like
        CO2 Stomatal conductance (mmol C02 m-2 s-1)
    assim : float or array_like
        Net Assimilation rate (micromol m-2 s-1)
    ci : float or array_like
        Leaf substomatal C02 concentration (micromol mol-1)

    References
    ----------
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    """

    [ca, vc_max, j_i, tes_star] = map(_check_default_parameter_size,
                                      [ca, vc_max, j_i, tes_star], 4 * [vpd])

    # Compensation point
    tes = compensation_point(tes_star, vc_max, rd, kc, ko, oi=oi)

    # parameters for temp limitation (Eq. A1 in [Leuning_1995]_)
    bb_t = kc * (1. + oi / ko)
    bb_r = 8. * tes_star

    # Initialize arrays
    ci_eq = (j_i * bb_t - vc_max * bb_r) / (vc_max * 4. - j_i * 1.)
    ci_sol = np.full(vpd.shape, FILL_DATA)
    ci_prev = np.full(vpd.shape, FILL_DATA)
    w = np.zeros(ca.shape)
    # Initialize diff_prev as NaN to force at least one iteration
    diff_prev = np.full(vpd.shape, np.nan)

    for ci in np.linspace(0.98 * ca, 0.1 * ca, N_ELEMENTS_CI):
        temp = ci < ci_eq
        # Assimilation temp limited(mol/ m2/s)
        # Rubisco-limited carboxylation rate
        w[temp] = w_func(ci[temp], tes_star[temp], vc_max[temp], bb_t[temp], 1.)
        # Assimilation rad limited(mol/ m2/s)
        # RuBP regeneration-limited carboxylation rate
        rad = ~temp
        w[rad] = w_func(ci[rad], tes_star[rad], j_i[rad], bb_r[rad], 4.)

        gs, assim = gs_leuning(rd, w, ca, ci, g0p, upper_lim=0.8)
        gs2 = gs_ball_berry_leuning(assim, ca, vpd, tes=tes, g0p=g0p, a_1=a_1, d_0=d_0)
        gs2 /= GV_GC_RATIO
        diff = gs - gs2
        abs_diff = np.abs(diff)
        j = np.logical_and(ci < 0.98 * ca, np.sign(diff) != np.sign(diff_prev))
        ci_sol[j] = update_value(ci[j], ci_prev[j], diff[j], diff_prev[j])
        ci_prev = ci.copy()
        diff_prev = diff.copy()
        if np.all(j):
            break

    non_valid = ci_sol == FILL_DATA
    diff_prev = np.full(non_valid.shape, np.nan)

    for ci in np.linspace(1.01 * ca, 5 * ca, N_ELEMENTS_CI):
        temp = np.logical_and(non_valid, ci < ci_eq)
        # Assimilation temp limited(mol/ m2/s)
        # Rubisco-limited carboxylation rate
        w[temp] = w_func(ci[temp], tes_star[temp], vc_max[temp], bb_t[temp], 1.)
        # Assimilation rad limited(mol/ m2/s)
        # RuBP regeneration-limited carboxylation rate
        rad = ~temp
        w[rad] = w_func(ci[rad], tes_star[rad], j_i[rad], bb_r[rad], 4.)

        gs, assim = gs_leuning(rd, w, ca, ci, g0p, upper_lim=0.8)
        gs2 = gs_ball_berry_leuning(assim, ca, vpd, tes=tes, g0p=g0p, a_1=a_1, d_0=d_0)
        gs2 /= GV_GC_RATIO
        diff = gs - gs2
        abs_diff = np.abs(diff)
        j = np.logical_and.reduce((non_valid, ci > 1.02 * ca,
                                   np.sign(diff) != np.sign(diff_prev)))
        ci_sol[j] = update_value(ci[j], ci_prev[j], diff[j], diff_prev[j])
        ci_prev = ci.copy()
        diff_prev = diff.copy()
        if np.all(j):
            break

    temp = ci_sol < ci_eq
    # Assimilation temp limited(mol/ m2/s)
    # Rubisco-limited carboxylation rate
    w[temp] = w_func(ci_sol[temp], tes_star[temp], vc_max[temp], bb_t[temp], 1.)
    # Assimilation rad limited(mol/ m2/s)
    # RuBP regeneration-limited carboxylation rate
    rad = ~temp
    w[rad] = w_func(ci_sol[rad], tes_star[rad], j_i[rad], bb_r[rad], 4.)
    gs, assim = gs_leuning(rd, w, ca, ci_sol, g0p, upper_lim=0.8)
    return gs, assim, ci_sol



def gs_ball_berry(assim, ca, h, g0p, a_1):
    """
    Parameters
    ----------
    assim : float or array_like
        Net Assimilation rate (micromol m-2 s-1)
    ca : float or array_like
        Air CO2 concentration (micromol mol-1)
    h : float or array_like
        relative humididy (unitless)
    g0p : float
        Conductance at night (mol m-2 s-1)
    a_1 : float
        Conversion factor Leuning gs model

    Returns
    -------
    gs : float or array_like
        CO2 Stomatal conductance (mol/m2 s)

    References
    ----------

    """
    gs = g0p + a_1 * assim * h / ca
    return gs


def gs_ball_berry_leuning(assim,
                          ca,
                          vpd,
                          tes=0,
                          g0p=G0P,
                          a_1=A_GS,
                          d_0=D_0_GS,
                          fw=1):
    """
    Parameters
    ----------
    assim : float or array_like
        Net Assimilation rate (micromol m-2 s-1)
    ca : float or array_like
        Air CO2 concentration (micromol mol-1)
    tes : float or array_like
        CO2 compensation point (micromol mol-1)
    vpd : float or array_like
        Vapor pressure deficit (mb)
    g0p : float
        H2O Conductance at night (mmol m-2 s-1)
    a_1 : float
        Conversion factor Leuning gs model
    d_0 : float
        Sensitivity of the stomata to VPD (mb)(Leuning,1995)
    fw : float or array_like
        Stress reduction factor of stomata conductance, default=1 (no stress)

    Returns
    -------
    gs : float or array_like
        H2O Stomatal conductance (mol/m2 s)

    References
    ----------
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    """
    # Eq. 8 in [Leuning1995]_
    gs = g0p + fw * (a_1 * assim) / ((ca - tes) * (1. + vpd / d_0))
    return gs


def gs_optimal_medlyn(assim, ca, vpd, g_1=41.6, g_0=0):
    """
    Parameters
    ----------
    assim : float or array_like
        Net Assimilation rate (micromol m-2 s-1)
    ca : float or array_like
        Air CO2 concentration (micromol mol-1)
    vpd : float or array_like
        Vapor pressure deficit (mb)
    g_1 : float
        slope parameter (mb**0.5), is inversely proportional to the square root of
        the carbon cost per unit water used by the plant
    g_0 : float
        Minimum or cuticular conductance at night (mmol m-2 s-1)

    Returns
    -------
    gs : float or array_like
        CO2 Stomatal conductance (mol/m2 s)

    References
    ----------
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    """
    # Eq. 11 in [Medlyn2011]_
    gs = (1. + g_1 / np.sqrt(vpd)) * assim / ca
    gs = np.maximum(g_0, gs)
    return gs


def w_func(ci, tes_star, ab, bb, eb):
    """
    Parameters
    ----------
    ci : float or array_like
        Leaf substomatal CO2 concentration (micromol mol-1)
    tes_star : float or array_like
        CO2 compensation point in the absence of dark respiration (micromol mol-1)
    ab : float or array_like
        Numerator parameter in photosyntesis model
    eb : float or array_like
        Multiplier parameter to ci in denominator of photosyntesis model
    bb : float or array_like
        Second term of denominator of photosyntesys model

    Returns
    -------
    a_q : float or array_like
        Assimilation rate

    References
    ----------
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    """
    # (Eq. A1 or A2 in [Leuning_1995]_)
    a_q = (ab * (ci - tes_star)) / (eb * ci + bb)
    return a_q


def gs_leuning(rd, w, ca, ci, g0p, upper_lim=0.8):
    """
    Parameters
    ----------
    rd : float or array_like
        Respiration (micromol m-2 s-1)
    w : float or array_like
        gross rate of photosynthesis (micromol m-2 s-1)
    ca : float or array_like
        Air C02 concentration (micromol mol-1)
    ci : float or array_like
        Leaf substomatal CO2 concentration (micromol mol-1)
    g0p : float
        Conductance at night (mmol m-2 s-1)
    upper_lim : float
        Maximum stomatal conductance (mmol m-2 s-1)

    Returns
    -------
    gs : float or array_like
        CO2 Stomatal conductance (mmol m-2 s-1)
    assim : float or array_like
        Net Assimilation rate  (micromol m-2 s-1)

    References
    ----------
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x
    """
    a = w - rd
    gs = np.clip(a / (ca - ci), g0p, upper_lim)  # Eq. 3a in [Leuning_1995]_
    # net assimilation (micromol/m2/s)
    assim = gs * (ca - ci)  # Eq. 3a in [Leuning_1995]_
    return gs, assim


def update_value(new_value, old_value, new_diff, old_diff):
    updated = new_value - new_diff * (old_value - new_value) / (
            old_diff - new_diff)
    return updated


def mmolh20_2_wm2(t_a):
    '''Calculates the conversion factor for transpiration rate from
    mmol H20 m^-2 s^-1 to Wm^-2
    to mol m-2 s-1.

    Parameters
    ----------
    t_a : float
        Air temperature (K).

    Returns
    -------
    k : float
        Conversion factor from  mmol m-2 s-1 to Wm-2.
    '''

    molar_weight = 18e-3  # H20 molar weight in kg

    lambda_v = met.calc_lambda(t_a)  # J kg-1
    k = 1e-3 * molar_weight * lambda_v
    return np.asarray(k)


def soil_respiration(t_s_k, f_soil_25=3., del_h=53e3):
    """Function to compute net assimilation and photosynthesis given canopy resistances
     without taking into account the stomatal conductance reduction
    associated with the leaf water potential

    Parameters
    ----------
    t_s_k : float or array_like
        Soil temperature (Kelvin)
    f_soil_25 : float or array_like
        CO2 flux from the soil at T0 = 298 K, default = 3 micromol m-2 s-1
    del_h : float or array_like
        Activation energy, default = 53000 J mol -1,

    Returns
    -------
    f_soil : float or array_like, optional
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).

    References
    ----------
    .. [Tuzet2003] Tuzet, A., Perrier, A., Leuning, R., 2003.
        A coupled model of stomatal conductance, photosynthesis and transpiration.
        Plant, Cell & Environment 26, 1097–1116.
        https://doi.org/10.1046/j.1365-3040.2003.01035.x
    """
    # Eq. A12 in [Tuzet2003]_
    f_soil = f_soil_25 * np.exp(del_h * (t_s_k - 298.15) / (GAS_CONSTANT * t_s_k * 298.15))
    return f_soil


def soil_respiration_lloyd(t_s_k, f_soil_ref=2., e_0=308.56, t_ref=283.15):
    """Function to compute net assimilation and photosynthesis given canopy resistances
     without taking into account the stomatal conductance reduction
    associated with the leaf water potential

    Parameters
    ----------
    t_s_k : float or array_like
        Soil temperature (Kelvin)
    f_soil_ref : float or array_like
        CO2 flux from the soil at reference temperature,
        default = 3 micromol m-2 s-1 at 283.15K
    e_0 : float or array_like
        Temperature sensitivity (Kelvin), default = 308.56K
    t_ref : float or array_like
        Corresponding temperature reference (Kelvin) for f_soil_ref

    Returns
    -------
    f_soil : float or array_like, optional
        Soil Respiration (micromol m-2 s-1), default = 0 (micromol m-2 s-1).

    References
    ----------
    .. [Lloyd1994] Lloyd, J., Taylor, J.A., 1994.
        On the Temperature Dependence of Soil Respiration.
        Functional Ecology 8, 315.
        https://doi.org/10.2307/2389824
    """
    # Eq. 11 in [Lloyd1994]_
    f_soil = f_soil_ref * np.exp(e_0 * (1. / (t_ref - 227.13) - 1. / (t_s_k - 227.13)))
    return f_soil


def compensation_point(tes_star, vc_max, rd, kc, ko, oi=OI):
    k_prime = kc * (1. + oi / ko)
    tes = (tes_star * vc_max + rd * k_prime) / (vc_max - rd)
    return tes


def get_photosynthesis_params(t_k,
                              c_kc=None,
                              c_ko=None,
                              c_tes=None,
                              c_rd=None,
                              c_vcx=None,
                              c_jx=None,
                              c_tpu=None):
    """

    Parameters
    ----------
    t_k : float or array
        Temperature (Kelvin)
    c_kc : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for CO2 (no units).
        Default (38.05, 79430 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_ko : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        Michaelis constant for O2 (no units).
        Default (20.30, 36380 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_tes : tuple
        Reference value at 25ªC and activation energy (J mol–1) for
        CO2 compensation point in the absence of Rd (micromol mol-1)
        Default (19.02, 37830 J mol-1) after [Bernacchi2001]_ for C3 plants
    c_rd : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        respiration (micromol m-2 s-1).
        Default (17.91, 44790 J mol-1) after [DiazEspejo2006]_ for olives
    c_vcx : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum catalytic activity of Rubisco in the presence of saturating
        amounts of RuBP and CO2 (micromol m-2 s-1).
        Default (33.99, 73680 J mol-1) after [DiazEspejo2006]_ for olives
    c_jx : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum rate of electron transport at saturating irradiance (micromol m-2 s-1)
        Default (18.88, 35350 J mol-1) after [DiazEspejo2006]_ for olives
    c_tpu : tuple
        Reference value at 25ªC  and activation energy (J mol–1) for
        maximum rate of Triose phosphate utilization (micromol m-2 s-1)
        Default (18.88, 35350 J mol-1) after [DiazEspejo2006]_ for olives
    Returns
    -------

    """
    if c_kc is None:
        c_kc = DEFAULT_C_KC
    if c_ko is None:
        c_ko = DEFAULT_C_KO
    if c_tes is None:
        c_tes = DEFAULT_C_TES
    if c_rd is None:
        c_rd = DEFAULT_C_RD
    if c_vcx is None:
        c_vcx = DEFAULT_C_VCX
    if c_jx is None:
        c_jx = DEFAULT_C_JX
    if c_tpu is None:
        c_tpu = DEFAULT_C_TPU

    tes_star = t_func_arrhenius(t_k, *c_tes)  # CO2 compensation point in the absence of Rd  (micromol mol-1)
    kc = t_func_arrhenius(t_k, *c_kc)  # Michaelis constant for CO2 (no units)
    ko = t_func_arrhenius(t_k, *c_ko)  # Michaelis constant for O2 (no units)
    # maximum catalytic activity of Rubisco in the presence of saturating amounts
    # of RuBP and CO2 (micromol m-2 s-1)
    vc_max = t_func_arrhenius(t_k, *c_vcx)  # Arrhenius temperature function
    # maximum rate of electron transport at saturating irradiance (micromol m-2 s-1)
    j_max = t_func_arrhenius(t_k, *c_jx)  # Arrhenius temperature function
    # "Dark" respiration (micromol m-2 s-1)
    rd = t_func_arrhenius(t_k, *c_rd)  # Arrhenius temperature function
    ## rate of Pi release associated with triose phosphate utilization ()
    tpu = t_func_arrhenius(t_k, *c_tpu)  # Arrhenius temperature function

    return vc_max, j_max, rd, kc, ko, tes_star, tpu


def electron_transport_rate(apar, j_max, alpha=0.20, theta=0.95):
    if theta:
        j = j_hyperbolic(apar, j_max, alpha=alpha, theta=theta)
    else:
        j = j_voncaemmerer(apar, j_max, alpha=alpha)

    return j

def j_hyperbolic(apar, j_max, alpha=0.20, theta=0.95):
    """Compute electron transport rate based on [Farquhar_1984]_ non-rectangular
    hyperbolic function.

    $\theta J^2  - ( \alpha Q + J_{lmax} ) J + \alpha Q J_{lmax} = 0$

    Parameters
    ----------
    apar : float or array_like
        Absorbed photon flux density (micromol m-2 s-1)
    j_max : float or array_like
        Maximum electron transport rate (micromol m-2 s-1)
    alpha : float or array_like
        Quantum yield (mol electrons / mol quanta).
        Default 0.2 from [Leuning1995]_
    theta : float or array_like
        curvature of leaf response of electron transport to irradiance.
        Default 0.95 from [Leuning1995]_

    Returns
    -------
    j : float or array_like
        Electron transport rate (micromol m-2 s-1)

    References
    ---------.
    .. [Farquhar_1984] Farquhar, G.D. and Wong, S.C.,
        An empirical model of stomatal conductance. (1984)
        Functional Plant Biology, 11(3), pp.191-210.
        https://doi.org/10.1071/PP9840191
    .. [Leuning_1995] Leuning. A critical appraisal of a combined stomatal-
        photosynthesis model for C3 plants.
        Plant, Cell and Environment (1995) 18, 339-355
        https://doi.org/10.1111/j.1365-3040.1995.tb00370.x

    """
    b = -(alpha * apar + j_max)
    c = alpha * apar * j_max
    j = (-b - np.sqrt(b ** 2 - 4. * theta * c)) / (2. * theta)
    return j

def j_voncaemmerer(apar, j_max, alpha=0.20):
    """Compute electron transport rate based on [vonCaemerer_1981]_ function.

    $\theta J^2  - ( \alpha Q + J_{lmax} ) J + \alpha Q J_{lmax} = 0$

    Parameters
    ----------
    apar : float or array_like
        Absorbed photon flux density (micromol m-2 s-1)
    j_max : float or array_like
        Maximum electron transport rate (micromol m-2 s-1)
    alpha : float or array_like
        Quantum yield (mol electrons / mol quanta).
        Default 0.2 from [Leuning1995]_

    Returns
    -------
    j : float or array_like
        Electron transport rate (micromol m-2 s-1)

    References
    ---------.
    .. [vonCaemerer_1981] Von Caemmerer, S. V., & Farquhar, G. D.
        Some relationships between the biochemistry of photosynthesis and
        the gas exchange of leaves. .
        Planta (1981), 153(4), 376-387.
        https://doi.org/10.1007/BF00384257
    """
    j = alpha * j_max * apar / (alpha * j_max + 2.1 * apar)
    return j

