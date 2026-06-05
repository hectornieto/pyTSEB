# TSEB Models Return Values Summary

## Overview
This document describes the correct return values for each TSEB model function in the **pyTSEB-master** folder.

## Return Values by Model

### TSEB_PT (Priestley-Taylor TSEB)
**Returns 19 values:**
1. flag
2. T_S (Soil temperature in Kelvin)
3. T_C (Canopy temperature in Kelvin)
4. T_AC (Air temperature at canopy interface in Kelvin)
5. Ln_S (Soil net longwave radiation in W/m²)
6. Ln_C (Canopy net longwave radiation in W/m²)
7. LE_C (Canopy latent heat flux in W/m²)
8. H_C (Canopy sensible heat flux in W/m²)
9. LE_S (Soil latent heat flux in W/m²)
10. H_S (Soil sensible heat flux in W/m²)
11. G (Soil heat flux in W/m²)
12. R_S (Soil aerodynamic resistance to heat transport in s/m)
13. R_x (Bulk canopy aerodynamic resistance to heat transport in s/m)
14. R_A (Aerodynamic resistance to heat transport in s/m)
15. u_friction (Friction velocity in m/s)
16. L (Monin-Obukhov length in m)
17. n_iterations (Number of iterations until convergence)
18. **alpha_PT_rec** (Recommended/actual Priestley-Taylor coefficient used)

### TSEB_SW (Shuttleworth-Wallace TSEB)
**Returns 20 values:**
1. flag
2. T_S (Soil temperature in Kelvin)
3. T_C (Canopy temperature in Kelvin)
4. T_AC (Air temperature at canopy interface in Kelvin)
5. Ln_S (Soil net longwave radiation in W/m²)
6. Ln_C (Canopy net longwave radiation in W/m²)
7. LE_C (Canopy latent heat flux in W/m²)
8. H_C (Canopy sensible heat flux in W/m²)
9. LE_S (Soil latent heat flux in W/m²)
10. H_S (Soil sensible heat flux in W/m²)
11. G (Soil heat flux in W/m²)
12. R_S (Soil aerodynamic resistance to heat transport in s/m)
13. R_x (Bulk canopy aerodynamic resistance to heat transport in s/m)
14. R_A (Aerodynamic resistance to heat transport in s/m)
15. Rss_out (Soil surface resistance to water vapor transport in s/m)
16. Rst_out (Stomatal resistance to water vapor transport in s/m)
17. **R_c** (Canopy resistance to water vapor transport in s/m)
18. u_friction (Friction velocity in m/s)
19. L (Monin-Obukhov length in m)
20. n_iterations (Number of iterations until convergence)

### TSEB_PM (Penman-Monteith TSEB)
**Returns 18 values:**
1. flag
2. T_S (Soil temperature in Kelvin)
3. T_C (Canopy temperature in Kelvin)
4. T_AC (Air temperature at canopy interface in Kelvin)
5. Ln_S (Soil net longwave radiation in W/m²)
6. Ln_C (Canopy net longwave radiation in W/m²)
7. LE_C (Canopy latent heat flux in W/m²)
8. H_C (Canopy sensible heat flux in W/m²)
9. LE_S (Soil latent heat flux in W/m²)
10. H_S (Soil sensible heat flux in W/m²)
11. G (Soil heat flux in W/m²)
12. R_S (Soil aerodynamic resistance to heat transport in s/m)
13. R_x (Bulk canopy aerodynamic resistance to heat transport in s/m)
14. R_A (Aerodynamic resistance to heat transport in s/m)
15. u_friction (Friction velocity in m/s)
16. L (Monin-Obukhov length in m)
17. n_iterations (Number of iterations until convergence)
18. **R_c** (Canopy resistance to water vapor transport in s/m)

## Key Differences

1. **TSEB_PT** is the only model that returns `alpha_PT_rec`
2. **TSEB_SW** is the only model that returns both `Rss_out` and `Rst_out` in addition to `R_c`
3. **TSEB_PM** only returns `R_c` among the resistance outputs (no Rss_out, Rst_out)
4. All models share the same first 14 outputs in the same order

## Important Notes

- The code in `PT and SW.py` is **correctly configured** for these return values
- Error handling with traceback has been added for better debugging
- Make sure to run the script from within the `pyTSEB-master` directory to ensure correct imports

## Usage

Run the script:
```bash
cd pyTSEB-master
python "PT and SW.py"
```

The script will:
1. Run all three TSEB models with sample data
2. Save results to Excel file: `tseb_comparison_results_3models.xlsx`
3. Display any errors with full traceback information

