.. pyTSEB documentation master file, created by
   sphinx-quickstart on Tue Jan 26 10:54:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyTSEB's documentation!
==================================

Summary
-------
The Two Source Energy Balance model computes the turbulent fluxes for two layers, soil and vegetation, where both layers are allow to interact between them. Therefore allowing an analogy between the flux transport and the *Ohm's Law* for transport of electricity, the sensible *H* and latent :math:`\lambda E` heat fluxes can be represented as in 

.. image:: TSEB_Scheme.png
	:alt: TSEB resistance network in series

where :math:`\lambda E` is estimated as a residual of the surface energy balance.

.. math::
	\lambda{}E_{S}&\approx R_{n,S}-G-H_{S}\\
	\lambda{}E_{C}&\approx R_{n,C}-H_{C}


with
 
.. math::
	H&=H_C+H_S\\
	&=\rho_{air} C_{p}\frac{T_{AC}-T_{A}}{r_{a}}\\
	H_{C}&=\rho_{air} C_{p}\frac{T_{C}-T_{AC}}{r_{x}}\\
	H_{S}&=\rho_{air} C_{p}\frac{T_{S}-T_{AC}}{r_{s}}


where :math:`\rho_{air}` is the density of air, :math:`C_{p}` is the heat capacity of air at constant pressure, :math:`T_{AC}` is the air temperature at the canopy interface:

.. math::
	T_{AC}=\frac{\frac{T_A}{r_a}+\frac{T_C}{r_x}+\frac{T_S}{r_s}}{\frac{1}{r_a}+\frac{1}{r_x}+\frac{1}{r_s}}


:math:`T_C` and :math:`T_S` are related to the directional radiometric temperature :math:`T_{rad}\left(\theta\right)` by

.. math::
	\sigma T_{rad}^4\left(\theta\right)=f_c\left(\theta\right)\sigma\,T_{C}^4+\left[1-f_{c}\left(\theta\right)\right]\sigma\,T_{S}^4


with :math:`f_c\left(\theta\right)` is the fraction of vegetation observed by a sensor pointing at a zenith angle :math:`\theta`

.. math::
	f_c\left(\theta\right)=1-\exp\left[-\kappa_{be}\left(\theta\right)\mathrm{LAI}\right]


and :math:`\kappa_{be}\left(\theta\right)=\frac{\sqrt{\chi^2+\tan^2\theta}}{\chi+1.774\left(\chi+1.182\right)^{-0.733}}` is the extintion coefficient of a canopy with a leaf angle distribution function defined by the Cambpell 1990 :math:`\chi` parameter

Contents
--------

.. toctree::

	pyTSEB
	README_Notebooks
	TSEB
	netRadiation
	ClumpingIndex
	MOsimilarity
	resistances
	meteoUtils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

