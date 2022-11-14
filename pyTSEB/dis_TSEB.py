#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:10:34 2018

@author: Hector Nieto (hector.nieto@ica.csic.es)
"""
import numpy as np
from osgeo import gdal

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter

from . import TSEB

# ==============================================================================
# List of constants used in dis_TSEB model and sub-routines
# ==============================================================================
ITERATIONS_OUT = 50
DIS_TSEB_ITERATIONS = 50
NO_VALID_FLAG = 255
VALID_FLAG = 0


def dis_TSEB(flux_LR,
             scale,
             Tr_K,
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
             UseL=np.inf,
             leaf_width=0.1,
             z0_soil=0.01,
             alpha_PT=1.26,
             x_LAD=1,
             f_c=1.0,
             f_g=1.0,
             w_C=1.0,
             resistance_form=[0, {}],
             calcG_params=[[1], 0.35],
             massman_profile=[0, []],
             flux_LR_method='EF',
             correct_LST=True):

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
    UseL : float or None, optional
        Its value will be used to force the Moning-Obukhov stability length.
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

            * [1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).

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


    # Initialize HR output variables
    [flag,
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
     n_iterations] = map(np.empty, 17*[Tr_K.shape])

    [T_S[:],
     T_C[:],
     T_AC[:],
     Ln_S[:],
     Ln_C[:],
     LE_C[:],
     H_C[:],
     LE_S[:],
     H_S[:],
     G[:],
     R_S[:],
     R_x[:],
     R_A[:],
     u_friction[:],
     L[:]] = 15*[np.nan]

    n_iterations[:] = 0
    flag[:] = NO_VALID_FLAG

    gt_LR = scale[0]
    prj_LR = scale[1]
    gt_HR = scale[2]
    prj_HR = scale[3]

    # Create mask that masks high-res pixels where low-res constant ratio
    # does not exist or is invalid
    dims_LR = flux_LR.shape
    dims_HR = Tr_K.shape
    const_ratio = scale_with_gdalwarp(flux_LR, prj_LR, prj_HR, dims_HR, gt_LR, gt_HR,
                                      gdal.GRA_NearestNeighbour)
    mask = np.ones(const_ratio.shape, dtype=bool)
    mask[np.logical_or(np.isnan(const_ratio), Tr_K <= 0)] = False

    # Set the starting conditions for disaggregation.
    counter = np.ones(const_ratio.shape)
    counter[~mask] = np.nan
    T_offset = np.zeros(const_ratio.shape)
    T_offset[~mask] = np.nan
    Tr_K_modified = Tr_K.copy()
    T_A_K_modified = T_A_K.copy()

    const_ratio_diff = np.zeros(const_ratio.shape)+1000
    const_ratio_HR = np.ones(const_ratio.shape)*np.nan

    print('Forcing low resolution MO stability length as starting point in the iteration')
    if isinstance(UseL, float):
        L = np.ones(Tr_K.shape) * UseL
    else:
        L = scale_with_gdalwarp(UseL, prj_LR, prj_HR, dims_HR, gt_LR, gt_HR,
                                gdal.GRA_NearestNeighbour)
    del UseL

    rho = TSEB.met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = TSEB.met.calc_c_p(p, ea)  # Heat capacity of air

    #######################################################################
    # For all the pixels in the high res. TSEB
    # WHILE high-res contant ration != low-res constant ratio
    #   adjust Tair or LST for unmasked pixels
    #   run high-res TSBE for unmaksed pixels
    #   claculate high-res consant ratio
    #   mask pixels where ratios aggree
    while np.any(mask) and np.nanmax(counter) < DIS_TSEB_ITERATIONS:

        # Adjust LST or air temperature as required
        if correct_LST:
            Tr_K_modified[mask] = _adjust_temperature(Tr_K[mask], T_offset[mask], correct_LST,
                                                      flux_LR_method)
        else:
            T_A_K_modified[mask] = _adjust_temperature(T_A_K[mask], T_offset[mask], correct_LST,
                                                       flux_LR_method)

        # Run high-res TSEB on all unmasked pixels
        flag[mask] = VALID_FLAG

        # First process bare soil cases
        print('First process bare soil cases')
        i = np.array(np.logical_and(LAI == 0, mask))

        [flag[i],
         Ln_S[i],
         LE_S[i],
         H_S[i],
         G[i],
         R_A[i],
         u_friction[i],
         L[i],
         n_iterations[i]] = TSEB.OSEB(Tr_K_modified[i],
                                      T_A_K_modified[i],
                                      u[i],
                                      ea[i],
                                      p[i],
                                      Sn_S[i],
                                      L_dn[i],
                                      emis_S[i],
                                      z_0M[i],
                                      d_0[i],
                                      z_u[i],
                                      z_T[i],
                                      calcG_params=[calcG_params[0], calcG_params[1][i]],
                                      const_L=L[i])

        T_S[i] = Tr_K_modified[i]
        T_AC[i] = T_A_K_modified[i]
        # Set canopy fluxes to 0
        Sn_C[i] = 0.0
        Ln_C[i] = 0.0
        LE_C[i] = 0.0
        H_C[i] = 0.0

        # Then process vegetated pixels
        print('Then process vegetated pixels')
        i = np.array(np.logical_and(LAI > 0, mask))
        if resistance_form[0] == 0:
            resistance_flag = [resistance_form[0],
                               {k: resistance_form[1][k][i] for k in resistance_form[1]}]

        else:
            resistance_flag = [resistance_form[0], {}]

        [flag[i],
         T_S[i],
         T_C[i],
         T_AC[i],
         Ln_S[i],
         Ln_C[i],
         LE_C[i],
         H_C[i],
         LE_S[i],
         H_S[i],
         G[i],
         R_S[i],
         R_x[i],
         R_A[i],
         u_friction[i],
         L[i],
         n_iterations[i]] = TSEB.TSEB_PT(Tr_K_modified[i],
                                         vza[i],
                                         T_A_K_modified[i],
                                         u[i],
                                         ea[i],
                                         p[i],
                                         Sn_C[i],
                                         Sn_S[i],
                                         L_dn[i],
                                         LAI[i],
                                         h_C[i],
                                         emis_C[i],
                                         emis_S[i],
                                         z_0M[i],
                                         d_0[i],
                                         z_u[i],
                                         z_T[i],
                                         leaf_width=leaf_width[i],
                                         z0_soil=z0_soil[i],
                                         alpha_PT=alpha_PT[i],
                                         x_LAD=x_LAD[i],
                                         f_c=f_c[i],
                                         f_g=f_g[i],
                                         w_C=w_C[i],
                                         resistance_form=resistance_flag,
                                         calcG_params=[calcG_params[0], calcG_params[1][i]],
                                         const_L=L[i])

        LE_HR = LE_C + LE_S
        H_HR = H_C + H_S

        print('Recalculating MO stability length')
        L = TSEB.MO.calc_L(u_friction, T_A_K_modified, rho, c_p, H_HR, LE_HR)

        # Calcualte HR constant ratio
        valid = np.logical_and(mask, flag != NO_VALID_FLAG)
        if flux_LR_method == 'EF':
            # Calculate high-res Evaporative Fraction
            const_ratio_HR[valid] = LE_HR[valid] / (LE_HR[valid] + H_HR[valid])
        elif flux_LR_method == 'LE':
            # Calculate high-res Evaporative Fraction
            const_ratio_HR[valid] = LE_HR[valid]
        elif flux_LR_method == 'H':
            # Calculate high-res Evaporative Fraction
            const_ratio_HR[valid] = H_HR[valid]

        # Calculate average constant ratio for each LR pixel from all HR
        # pixels it contains
        print('Calculating average constant ratio for each LR pixel using valid HR pixels')
        const_ratio_LR = scale_with_gdalwarp(const_ratio_HR, prj_HR, prj_LR, dims_LR, gt_HR, gt_LR,
                                             gdal.GRA_Average)
        const_ratio_HR = scale_with_gdalwarp(const_ratio_LR, prj_LR, prj_HR, dims_HR, gt_LR, gt_HR,
                                             gdal.GRA_NearestNeighbour)
        const_ratio_HR[~mask] = np.nan

        # Mask the low-res pixels for which constant ratio of hig-res and
        # low-res runs agree.
        const_ratio_diff = const_ratio_HR - const_ratio
        const_ratio_diff[np.logical_or(np.isnan(const_ratio_HR),
                                       np.isnan(const_ratio))] = 0

        # Calculate temperature offset and ready-pixels mask
        if flux_LR_method == 'EF':
            mask = np.abs(const_ratio_diff) > 0.01
            step = np.clip(const_ratio_diff*5, -1, 1)
        elif flux_LR_method == 'LE' or flux_LR_method == 'H':
            mask = np.abs(const_ratio_diff) > 5
            step = np.clip(const_ratio_diff*0.01, -1, 1)
        counter[mask] += 1
        T_offset[mask] += step[mask]

        print('disTSEB iteration %s' % np.nanmax(counter))
        print('Recalculating over %s high resolution pixels' % np.size(Tr_K[mask]))

    ####################################################################
    # When constant ratios for all pixels match, smooth the resulting Ta adjustment
    # with a moving window size of 2x2 km and perform a final run of high-res model
    mask = np.ones(const_ratio.shape, dtype=bool)
    mask[np.isnan(const_ratio)] = False

    T_offset_orig = T_offset.copy()
    T_offset = moving_gaussian_filter(T_offset, int(2000/float(gt_HR[1])))

    # Smooth MO length
    L = moving_gaussian_filter(L, int(2000/float(gt_HR[1])))

    if correct_LST:
        Tr_K_modified = Tr_K.copy()
        Tr_K_modified[mask] = _adjust_temperature(Tr_K[mask], T_offset[mask], correct_LST,
                                                  flux_LR_method)
    else:
        T_A_K_modified = T_A_K.copy()
        T_A_K_modified[mask] = _adjust_temperature(T_A_K[mask], T_offset[mask], correct_LST,
                                                   flux_LR_method)

    flag[mask] = VALID_FLAG

    # Run high-res TSEB on all unmasked pixels
    TSEB.ITERATIONS = ITERATIONS_OUT
    print('Final run of TSEB at high resolution with adjusted temperature')

    # First process bare soil cases
    print('First process bare soil cases')
    i = np.array(np.logical_and(LAI == 0, mask))
    [flag[i],
     Ln_S[i],
     LE_S[i],
     H_S[i],
     G[i],
     R_A[i],
     u_friction[i],
     L[i],
     n_iterations[i]] = TSEB.OSEB(Tr_K_modified[i],
                                  T_A_K_modified[i],
                                  u[i],
                                  ea[i],
                                  p[i],
                                  Sn_S[i],
                                  L_dn[i],
                                  emis_S[i],
                                  z_0M[i],
                                  d_0[i],
                                  z_u[i],
                                  z_T[i],
                                  calcG_params=[calcG_params[0], calcG_params[1][i]],
                                  const_L=L[i])

    T_S[i] = Tr_K_modified[i]
    T_AC[i] = T_A_K_modified[i]
    # Set canopy fluxes to 0
    Sn_C[i] = 0.0
    Ln_C[i] = 0.0
    LE_C[i] = 0.0
    H_C[i] = 0.0

    # Then process vegetated pixels
    print('Then process vegetated pixels')
    i = np.array(np.logical_and(LAI > 0, mask))

    if resistance_form[0] == 0:
        resistance_flag = [resistance_form[0],
                           {k: resistance_form[1][k][i] for k in resistance_form[1]}]

    else:
        resistance_flag = [resistance_form[0], {}]

    [flag[i],
     T_S[i],
     T_C[i],
     T_AC[i],
     Ln_S[i],
     Ln_C[i],
     LE_C[i],
     H_C[i],
     LE_S[i],
     H_S[i],
     G[i],
     R_S[i],
     R_x[i],
     R_A[i],
     u_friction[i],
     L[i],
     n_iterations[i]] = TSEB.TSEB_PT(Tr_K_modified[i],
                                     vza[i],
                                     T_A_K_modified[i],
                                     u[i],
                                     ea[i],
                                     p[i],
                                     Sn_C[i],
                                     Sn_S[i],
                                     L_dn[i],
                                     LAI[i],
                                     h_C[i],
                                     emis_C[i],
                                     emis_S[i],
                                     z_0M[i],
                                     d_0[i],
                                     z_u[i],
                                     z_T[i],
                                     leaf_width=leaf_width[i],
                                     z0_soil=z0_soil[i],
                                     alpha_PT=alpha_PT[i],
                                     x_LAD=x_LAD[i],
                                     f_c=f_c[i],
                                     f_g=f_g[i],
                                     w_C=w_C[i],
                                     resistance_form=resistance_flag,
                                     calcG_params=[calcG_params[0], calcG_params[1][i]],
                                     const_L=L[i])

    return [flag,
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
            n_iterations,
            T_offset,
            counter,
            T_offset_orig]


def _adjust_temperature(temperature, offset, correct_LST, method):
    # If high-res H is too high (offset is +ve) -> decrease LST to reduce H
    if correct_LST and method == "H":
        modified = temperature - offset
    # If high-res H is too high (offset is +ve) -> increase air temperature to reduce H
    elif not correct_LST and method == "H":
        modified = temperature + offset
    # If high-res LE is too high (offset is +ve) -> increase LST to reduce LE
    elif correct_LST and method in ["LE", "EF"]:
        modified = temperature + offset
    # If high-res LE is too high (offset is +ve) -> decrease air temperature to reduce LE
    else:
        modified = temperature - offset
    return modified


def moving_gaussian_filter(data, window):
    # 95% of contribution is from within the window
    sigma = window / 4.0

    V = data.copy()
    V[data != data] = 0
    VV = gaussian_filter(V, sigma)

    W = 0*data.copy() + 1
    W[data != data] = 0
    WW = gaussian_filter(W, sigma)

    return VV/WW


def moving_mean_filter(data, window):

    ''' window is a 2 element tuple with the moving window dimensions (rows, columns)'''
    kernel = np.ones(window)/np.prod(np.asarray(window))
    data = convolve2d(data, kernel, mode='same', boundary='symm')

    return data


def moving_mean_filter_2(data, window):

    ''' window is a 2 element tuple with the moving window dimensions (rows, columns)'''
    data = uniform_filter(data, size=window, mode='mirror')

    return data


def save_img(data, geotransform, proj, outPath, noDataValue=np.nan, fieldNames=[]):

    # Start the gdal driver for GeoTIFF
    if outPath == "MEM":
        driver = gdal.GetDriverByName("MEM")
        driverOpt = []

    shape = data.shape
    if len(shape) > 2:
        ds = driver.Create(outPath, shape[1], shape[0], shape[2], gdal.GDT_Float32, driverOpt)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        for i in range(shape[2]):
            ds.GetRasterBand(i+1).WriteArray(data[:, :, i])
            ds.GetRasterBand(i+1).SetNoDataValue(noDataValue)
    else:
        ds = driver.Create(outPath, shape[1], shape[0], 1, gdal.GDT_Float32, driverOpt)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(data)
        ds.GetRasterBand(1).SetNoDataValue(noDataValue)

    return ds


def scale_with_gdalwarp(array, prj_in, prj_out, dims_out, gt_in, gt_out, resample_alg):

    in_src = save_img(array, gt_in, prj_in, 'MEM', noDataValue=np.nan, fieldNames=[])
    # Get template projection, extent and resolution
    extent = [gt_out[0], gt_out[3]+gt_out[5]*dims_out[0],
              gt_out[0]+gt_out[1]*dims_out[1], gt_out[3]]

    # Resample with GDAL warp
    outDs = gdal.Warp('',
                      in_src,
                      dstSRS=prj_out,
                      xRes=gt_out[1],
                      yRes=gt_out[5],
                      outputBounds=extent,
                      resampleAlg=resample_alg,
                      format="MEM")

    array_out = outDs.GetRasterBand(1).ReadAsArray()

    return array_out
