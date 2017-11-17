# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:58:34 2017

@author: hnieto
"""

from pyTSEB import TSEB 
import numpy as np

# Leaf stomata distribution
AMPHISTOMATOUS = 2
HYPOSTOMATOUS = 1

#==============================================================================
# List of constants used in TSEB model and sub-routines
#==============================================================================
# Change threshold in  Monin-Obukhov lengh to stop the iterations
L_thres = 0.00001
# mimimun allowed friction velocity
u_friction_min = 0.01
# Maximum number of interations
ITERATIONS = 100
# kB coefficient
kB = 2.0

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
                    calcG_params=[
                        [1],
                        0.35],
                    Rst_min=100,
                    leaf_type=AMPHISTOMATOUS,
                    environmental_factors=None):
    
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
    [flag, Ln, LE, H, G, R_A, iterations] = [np.zeros(T_A_K.shape)+np.NaN for i in range(7)]
    
    # Calculate the general parameters
    rho_a = TSEB.met.calc_rho(p, ea, T_A_K)              # Air density
    Cp = TSEB.met.calc_c_p(p, ea)                        # Heat capacity of air
    delta=10.*TSEB.met.calc_delta_vapor_pressure(T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_=TSEB.met.calc_lambda(T_A_K)                     # latent heat of vaporization MJ kg-1
    psicr=TSEB.met.calc_psicr(p, lambda_)                     # Psicrometric constant (mb K-1)
    es=TSEB.met.calc_vapor_pressure(T_A_K)             # saturation water vapour pressure in mb
    
    rho_cp=rho_a*Cp
    vpd=es-ea

    # Calculate bulk stomatal conductance
    R_c=bulk_stomatal_conductance(LAI, 
                                  Rst_min, 
                                  leaf_type=leaf_type, 
                                  environmental_factors=environmental_factors)

    # iteration of the Monin-Obukhov length
    L = np.asarray(np.zeros(T_A_K.shape) + np.inf)
    u_friction = TSEB.MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(u_friction_min, u_friction))

    L_old = np.ones(T_A_K.shape)
    L_diff = np.asarray(np.ones(T_A_K.shape) * np.inf)
    z_0H = TSEB.res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport
    
    # Calculate Net radiation
    T_0_K=T_A_K #♦ asumme aerodynamic tempearture equals air temperature
    Ln = emis * L_dn - emis * TSEB.met.calc_stephan_boltzmann(T_0_K)
    Rn = np.asarray(Sn + Ln)

    # Compute Soil Heat Flux
    i = np.ones(Rn.shape, dtype=bool)
    G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn, i)
    
    for n_iterations in range(ITERATIONS):
   
        if np.all(L_diff < L_thres):
            print("Finished interation with a max. L diff: " + str(np.max(L_diff)))
            break
        
        print("Iteration " + str(n_iterations) +
              ", max. L diff: " + str(np.max(L_diff)))

        i = np.logical_and(L_diff >= L_thres, flag != 255)
        iterations[i] = n_iterations
        flag[i] = 0  

        # Calculate aerodynamic resistances                                        
        R_A[i]=TSEB.res.calc_R_A(z_T[i], u_friction[i], L[i], d_0[i], z_0H[i])
        R_eff=R_c[i]/R_A[i]
        
        # Apply Penman Monteith Combination equation
        LE[i]=(delta[i]*(Rn[i]-G[i])+rho_cp[i]*vpd[i]/R_A[i])/(delta[i]+psicr[i]*(1.+R_eff))
        H[i]=Rn[i]-G[i]-LE[i]
        # Now L can be recalculated and the difference between iterations
        # derived
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
        u_friction = np.asarray(np.maximum(u_friction_min, u_friction))

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
                         resistance_form=[0, {}],
                         calcG_params=[
                                [1],
                                0.35],
                         massman_profile=[0,[]],
                         leaf_type=AMPHISTOMATOUS,
                         environmental_factors=None):
    
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
                         w_C,
                         Rst_min,
                         R_ss,
                         calcG_params[1],
                         leaf_type],
                        [T_A_K] * 24)
    
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    
    # Create the output variables
    [flag, vpd_0, Ln_C, Ln_S, LE, H, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, 
     iterations] = [np.zeros(T_A_K.shape)+np.NaN for i in range(15)]
    
    
    # Calculate the general parameters
    rho_a = TSEB.met.calc_rho(p, ea, T_A_K)              # Air density
    Cp = TSEB.met.calc_c_p(p, ea)                        # Heat capacity of air
    delta=10.*TSEB.met.calc_delta_vapor_pressure(T_A_K)  # slope of saturation water vapour pressure in mb K-1
    lambda_=TSEB.met.calc_lambda(T_A_K)                     # latent heat of vaporization MJ kg-1
    psicr=TSEB.met.calc_psicr(p, lambda_)                     # Psicrometric constant (mb K-1)
    es=TSEB.met.calc_vapor_pressure(T_A_K)             # saturation water vapour pressure in mb
    
    # Calculate LAI dependent parameters for dataset where LAI > 0
    F = np.asarray(LAI / f_c)  # Real LAI
    
    rho_cp=rho_a*Cp
    vpd=es-ea

    # Calculate bulk stomatal conductance
    R_c=bulk_stomatal_conductance(LAI, Rst_min, leaf_type=leaf_type, environmental_factors=environmental_factors)

    F = np.asarray(LAI / f_c)  # Real LAI
    omega0=TSEB.CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    
    
    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    L = np.asarray(np.zeros(T_A_K.shape) + np.inf)
    u_friction = TSEB.MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(u_friction_min, u_friction))
    L_old = np.ones(T_A_K.shape)
    L_diff = np.asarray(np.ones(T_A_K.shape) * float('inf'))
 
    z_0H = TSEB.res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport
   
    # First assume that temperatures equals the Air Temperature
    T_C, T_S = np.array(T_A_K), np.array(T_A_K)
    emis_surf=f_c*emis_C+(1.-f_c)*emis_S
    Ln=emis_surf*(L_dn-TSEB.rad.sb*T_A_K**4)
    Ln_S=Ln*np.exp(-0.95 * LAI)
    Ln_C=Ln-Ln_S

    for n_iterations in range(ITERATIONS):
   
        if np.all(L_diff < L_thres):
            print("Finished interation with a max. L diff: " + str(np.max(L_diff)))
            break
        
        print("Iteration " + str(n_iterations) +
              ", max. L diff: " + str(np.max(L_diff)))

        i = np.logical_and(L_diff >= L_thres, flag != 255)
        iterations[i] = n_iterations
        flag[i] = 0  

        # Calculate aerodynamic resistances
        R_A_params = {"z_T": z_T[i], "u_friction": u_friction[i], "L": L[i], 
                      "d_0": d_0[i], "z_0H": z_0H[i]}
        params = {k: res_params[k][i] for k in res_params.keys()}
        R_x_params = {"u_friction": u_friction[i], "h_C": h_C[i], "d_0": d_0[i],
                      "z_0M": z_0M[i], "L": L[i],  "LAI": LAI[i], 
                      "leaf_width": leaf_width[i], "massman_profile": massman_profile,
                      "res_params": params}
        R_S_params = {"u_friction": u_friction[i], 'u':u[i], "h_C": h_C[i], "d_0": d_0[i],
                      "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i], 
                       "LAI": LAI[i], "leaf_width": leaf_width[i], 
                       "z0_soil": z0_soil[i], "z_u": z_u[i],  
                       "deltaT": T_S[i] - T_C[i], "massman_profile": massman_profile,
                       'u':u[i],'rho':rho_a[i], 'c_p':Cp[i], 'f_cover':f_c[i], 
                       'w_C':w_C[i],
                       "res_params": params}
        res_types = {"R_A": R_A_params, "R_x": R_x_params, "R_S": R_S_params}
        R_A[i], R_x[i], R_S[i] = TSEB.calc_resistances(resistance_form, res_types)

        _, _, _, C_s, C_c = calc_effective_resistances_SW(R_A[i], 
                                               R_x[i], 
                                               R_S[i], 
                                               R_c[i],
                                               R_ss[i],
                                               delta[i],
                                               psicr[i])
        # Calculate net longwave radiation with current values of T_C and T_S
        Ln_C[i], Ln_S[i] = TSEB.rad.calc_L_n_Kustas(
            T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i])
        Rn_C = Sn_C + Ln_C
        Rn_S = Sn_S + Ln_S
        Rn = Rn_C+Rn_S
        # Compute Soil Heat Flux Ratio
        G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn_S, i)
        
        # Eq. 12 in Shuttleworth & Wallace 1985
        PM_C = (delta[i]*(Rn[i]-G[i])+(rho_cp[i]*vpd[i]-delta[i]*R_x[i]*(Rn_S[i]-G[i]))/(R_A[i]+R_x[i]))/\
            (delta[i]+psicr[i]*(1.+R_c[i]/(R_A[i]+R_x[i])))
        # Eq. 13 in Shuttleworth & Wallace 1985
        PM_S = (delta[i]*(Rn[i]-G[i])+(rho_cp[i]*vpd[i]-delta[i]*R_S[i]*Rn_C[i])/(R_A[i]+R_S[i]))/\
            (delta[i]+psicr[i]*(1.+R_ss[i]/(R_A[i]+R_S[i])))
        # Eq. 11 in Shuttleworth & Wallace 1985
        LE[i] = C_c*PM_C+C_s*PM_S
        H[i] = Rn[i]-G[i]-LE[i]
        
        # Compute canopy and soil  fluxes
        #Vapor pressure deficit at canopy source height (mb) # Eq. 8 in Shuttleworth & Wallace 1985
        vpd_0[i]=vpd[i]+(delta[i]*(Rn[i]-G[i])-(delta[i]+psicr[i])*LE[i])*R_A[i]/(rho_cp[i])
        # Eq. 9 in Shuttleworth & Wallace 1985
        LE_S[i]=(delta[i]*(Rn_S[i]-G[i])+rho_cp[i]*vpd_0[i]/R_S[i])/\
            (delta[i]+psicr[i]*(1.+R_ss[i]/R_S[i]))  
        H_S[i]=Rn_S[i]-G[i]-LE_S[i]
        # Eq. 10 in Shuttleworth & Wallace 1985
        LE_C[i]=(delta[i]*Rn_C[i]+rho_cp[i]*vpd_0[i]/R_x[i])/\
            (delta[i]+psicr[i]*(1.+R_c[i]/R_x[i])) 
        H_C[i]=Rn_C[i]-LE_C[i]
        
        T_C[i]=calc_T(H_C[i], T_A_K[i], R_A[i]+R_x[i], rho_a[i], Cp[i])
        T_S[i]=calc_T(H_S[i], T_A_K[i], R_A[i]+R_S[i], rho_a[i], Cp[i])
        flag[np.logical_and(i,T_C<0)]=255
        flag[np.logical_and(i,T_S<0)]=255
        
        # Now L can be recalculated and the difference between iterations
        # derived
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
        u_friction = np.asarray(np.maximum(u_friction_min, u_friction))
    
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
        
def bulk_stomatal_conductance(LAI, Rst, leaf_type=AMPHISTOMATOUS, environmental_factors=None):
    f_vpd=1.
    f_temp=1.
    f_rad=1.
    f_stress=1.
    if type(environmental_factors)!=type(None):
        f_vpd=vpd_factor_Noilhan(environmental_factors[0], environmental_factors[1], g_vpd=0.025)
        f_temp=temp_factor_Noilhan(environmental_factors[0])
        if len(environmental_factors[0])>2:
            f_rad=environmental_factors[3]
            if len(environmental_factors[0])==4:
                f_stress=environmental_factors[4]

    factor=f_vpd*f_temp*f_rad*f_stress
    R_c=Rst/(leaf_type*LAI*factor)
    return np.asarray(R_c)

def vpd_factor_Noilhan(T_A_K, ea, g_vpd=0.025):
    es=TSEB.met.calc_vapor_pressure(T_A_K)
    f=1.-g_vpd*(es-ea)
    f=np.clip(f,0,1)
    return f

def temp_factor_Noilhan(T_A_K):
    f=1.-0.0016*(298-T_A_K)**2
    f=np.clip(f,0,1)
    return f
    

def calc_T(H, T_A, R, rho_a, Cp):
    
    T=T_A+H*R/(rho_a*Cp)
    return T
    

def calc_effective_resistances_SW(R_A, R_x, R_S, R_c, R_ss, delta, psicr):
    delta_psicr=delta+psicr
    
    R_a_SW=delta_psicr*R_A                              # Eq. 16 Shuttleworth and Wallace 1988
    R_s_SW=delta_psicr*R_S + psicr*R_ss                 # Eq. 17 Shuttleworth and Wallace 1988
    R_c_SW=delta_psicr*R_x + psicr*R_c                  # Eq. 18 Shuttleworth and Wallace 1988
    C_c=1./(1.+R_c_SW*R_a_SW/(R_s_SW*(R_c_SW+R_a_SW)))   # Eq. 14 Shuttleworth and Wallace 1988
    C_s=1./(1.+R_s_SW*R_a_SW/(R_c_SW*(R_s_SW+R_a_SW)))   # Eq. 15 Shuttleworth and Wallace 1988
    
    return R_a_SW, R_s_SW, R_c_SW, C_s, C_c


