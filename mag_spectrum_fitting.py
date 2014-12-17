'''
::AUTHOR::
Andrew Stocker


::Modified by:
Xiaosheng Huang
    
    
'''

#from __future__ import print_function


import sys
import argparse
import cStringIO


import loader as l
from plot_from_contours import *

from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import cPickle
from time import localtime, strftime, gmtime

from scipy.interpolate import interp1d

import matplotlib as mpl
import pickle
import sncosmo as snc


from itertools import izip
from loader import redden_fm, redden_pl, redden_pl2
from pprint import pprint
from scipy.interpolate import interp1d
from sys import argv
#from mag_spectrum_fitting import filter_features

### CONST ###
c = 3e18  # speed of light in A/sec.

## Configuration
PLOTS_PER_ROW = 6
N_BUCKETS = 20
RED_LAW = redden_fm

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15


## ind_min = np.abs(red_curve).argmin()
## f99wv = np.array([ref_wave[ind_min-1],ref_wave[ind_min],ref_wave[ind_min+1]])
## f99ebv = np.array([red_curve[ind_min-1],red_curve[ind_min],red_curve[ind_min+1]])
## V_wave = np.interp(0., f99ebv, f99wv).  To figure out the wavelength of the "zero" of the F99 law.  Using different phases, the value varies a bit (my estimate: 5413.5+/-0.1).
## Haven't looked at this too closely.  But probably because of numerical error, such as during the interpolation process.  It's interesting
## that, reading through the fm_unred.py, which AS and ZR have adopted, it's not obvious what this value should be.
## Here I have arbitrarily chosen the first phase to infer what this wavelength should be.  -XH 11/18/14
## AS previously used V_wave = 5417.2
V_wave = 5413.5


FrFlx2mag = 2.5/np.log(10)  #  =1.0857362


def ABmag_nu(flux, wave = None):
    
    '''The flux used here should be in per Hz.  If not, it's converted to flux per Hz.  See BM12'''
    if wave != None:
        flux = flux * wave**2/c

    return -2.5*np.log10(flux) - 48.6  ## Bessell & Murphy 2012.  Use 48.577 to perfectly match Vega, which has a V mag of 0.03.  But if AB mag is
                                              ## consistently used by me for single-wavelength mag, and by sncosmo for synthetic photometry (check) then
                                              ## using 48.6 is just fine.


def extract_wave_flux_var(SN_obs, N_BUCKETS = -1, mask = None, norm_meth = 'AVG', ebv = None, rv = None):

    '''
    Added Nov 25, 2014.
    
    takes in 2 spectral pickle files from loader.py, extracts, interpolates and converts, and finally returns
    normalized magntiudes and magnitude variances.
    
    '''

    calib_err_mag = SN_obs[2]

    ## if de-reddening is needed:
    if ebv != None:
        SN = snc.Spectrum(SN_obs[1].wave, redden_fm(SN_obs[1].wave, SN_obs[1].flux, ebv, rv), SN_obs[1].error)
    else:
        SN = SN_obs[1]

    SN_flux = SN.flux

    var = SN.error  # it's called error because that's one of the attributes defined in sncosmo



    if (SN_flux <= 0).any():
        print "In extract_wave_flux_var():"
        print "some flux values are not positive:", SN_flux[np.where(SN_flux <= 0)]
        print "These values will be rejected below as nan for the log."
        print "(But it's better to deal with the non-pos values before taking the log (even before interpolation).  Something to deal with later.)"
        print "\n\n\n"



    #flux_interp = interp1d(SN.wave, SN_flux)  # interp1d returns a function, which can be evaluated at any wavelength one would want.
                                                 # think of the two arrays supplied as the "training set".  So flux_interp() is a function.

    ## This is flux per frequency -- in order to calculate the AB magnitude -- see Bessell & Murphy eq 2 and eq A1; O'Donnell Astro 511 Lec 14.
    ## It doesn't seem to make any difference in terms of determining RV but it is the right way of doing things.  Magnitudes and colors calculated
    ## from these flux values should be directly comparable to AB mag's for broad bands, if that's what Andrew calculated for synthetic photometry.
    ## Does F99 assume a certain magnitude system?  Is that why Amanullah 2014 used Vega?  (Did they use Vega?  I think they did.
    #flux = flux_interp(ref_wave)#*(ref_wave**2)
    #var = interp1d(SN.wave, var)(ref_wave)  ####****-----------> This is not the right way to figure out the right variance for the interpolated spectrum.

    if mask != None:
        flux = flux[mask]  # Note: mask has the same length as mag_norm, and contains a bunch of 0's and 1's (the 0's are where the blocked features are).
                                   # This is a very pythonic way of doing things: even though mask doesn't specifiy the indices of the wavelengths that should
                                   # be blocked, the operation mag_norm[mask] does just that.  One can think of mask as providing a truth table that tells python
                                   # which of the elements in mag_norm to keep and which to discard.  Yes, it doesn't make sense at first sight since mask doesn't
                                   # contain indices.  But it does work, and is the pythonic way!  -XH 11/25/14.
        ref_wave = ref_wave[mask]
        var = var[mask]

    flux = SN_flux
    flux_per_Hz = flux * (SN.wave**2/c)
    mag_avg_flux = ABmag_nu(np.mean(flux_per_Hz))
    #flux_single_V = flux_interp(V_wave)#*(V_wave**2)




    ## convert flux, variance, and calibration error to magnitude space
    ## The spectral case:
    if N_BUCKETS < 0:
        mag_norm, mag_var, mag_V = flux2mag(flux_per_Hz, flux, SN.wave, mag_avg_flux, var, norm_meth = norm_meth)
        return_wave = SN.wave
        return_flux = SN_flux
        return_flux_var = var
        mag_V_var = interp1d(SN.wave, mag_var)(V_wave)  # This is obviously not correct -- though I don't think mag_V_var is used anywhere.
    ## The photometry case -- note here I don't block features, since that is not generally possible (or physical) to do.
    else:
 
        lo_wave = 3300.
        hi_wave = 9700.
        
        ## The following two statements are redundant: they are already in main() of photom_vs_spectral.py.
        filters_bucket, zp_bucket, LOW_wave, HIGH_wave = l.generate_buckets(lo_wave, hi_wave, N_BUCKETS)  #, inverse_microns=True)
        filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff for f in filters_bucket])
        
        return_wave = filter_eff_waves

        prefix = zp_bucket['prefix']  # This specifies units as inverse micron or angstrom; specified in the function call to l.generate_buckets().
        print 'filters_bucket', filters_bucket

        del_wave = (HIGH_wave  - LOW_wave)/N_BUCKETS
        
        band_flux = np.array([SN.bandflux(prefix+f, del_wave = del_wave, AB_nu = True)[0] for f in filters_bucket])  ## the element 1 give error, or it should be variance.  But check!  Since variance
                                                                                                                     ## and error for a band would be very different!
        return_flux = band_flux
        SN_mags = {f:ABmag_nu(band_flux[i]) for i, f in enumerate(filters_bucket)}
        
        mag_V =  SN_mags['V']
        
        print 'norm_meth', norm_meth    
        
        if norm_meth == 'AVG':
            mag_zp = mag_avg_flux
        elif norm_meth == 'V_band':
            mag_zp = mag_V
        elif norm_meth == None:
            mag_zp = 0.


        print 'mag_zp, mag_V, mag_avg_flux', mag_zp, mag_V, mag_avg_flux    


## THIS VERY EXPRESSION MAKES IT CLEAR WHY USING V BAND TO NORMALIZE IS A BAD IEAD: THE UNCERTAINTY OF MAG_NORM WILL THEN BE THE UNCERTAINTY OF SN_mag[f] and SN_mag['V'] (BOTH MEASUREMENT AND INTRINSIC
## DISPERSION, WHICH IS DIFFICULT TO ESTIMATE) ADDED IN QUARATRURE.  WHEREAS IF ONE USES mag_avg_flux, THE UNCERTAINTY WILL BE MUCH SMALLER -- IT IS IN FACT THE FLXERROR IN THE HEADER, ABOUT 0.02 MAG.

        mag_norm = -(np.array([SN_mags[f] for f in filters_bucket]) - mag_zp) ## the problem with doing things this way is using vs. not using norm_meth, the sign of SN_mag will be flipped.
                                                                       ## the minus sign is because we will plot E(V-X)

        return_flux_var = np.array([SN.bandflux(prefix+f, del_wave = del_wave, AB_nu = True)[1] for f in filters_bucket])



        ## calculate magnitude uncertainty
        ## note the extra factor of lambda*2/c actually gets canceled.
        fr_err = np.sqrt(return_flux_var)/band_flux



        mag_var = (FrFlx2mag*fr_err)**2

        ## To get variance for V band.  I'm sure there is a pythonic, list comprehension with a lambda function way of doing this.
        for i, f in enumerate(filters_bucket):
            if f == 'V':
                mag_V_var = mag_var[i]
                print 'V band fr_err', fr_err[i]  ## fractional uncertainty should improve as the number of bands decrease (but still above 5 bands.)
                print 'mag_V_var', mag_V_var
                    #                exit(1)

## Ignore the following line for now.  I'm NOT blocking any features.  12/7/14
#    if mask != None:
#        mag_norm = mag_norm[mask]  # Note: mask has the same length as mag_norm, and contains a bunch of 0's and 1's (the 0's are where the blocked features are).
#                                   # This is a very pythonic way of doing things: even though mask doesn't specifiy the indices of the wavelengths that should
#                                   # be blocked, the operation mag_norm[mask] does just that.  One can think of mask as providing a truth table that tells python
#                                   # which of the elements in mag_norm to keep and which to discard.  Yes, it doesn't make sense at first sight since mask doesn't
#                                   # contain indices.  But it does work, and is the pythonic way!  -XH 11/25/14.


    ## get mask for nan-values
    nanmask = ~np.isnan(mag_norm)
    

    return mag_norm, return_wave, return_flux, return_flux_var, mag_avg_flux, mag_V, mag_var, mag_V_var, calib_err_mag, nanmask, flux


def flux2mag(flux_per_Hz, flux, SN_wave, mag_avg_flux, var=None, norm_meth = 'AVG'):
    mag_var = None
    
    mag = ABmag_nu(flux_per_Hz)
    ## One shouldn't use the photon noise as the weight to find the average flux - see NB 11/22/14.
    ## Also note it's not the avg mag but the mag of the avg flux.

    ## The following two lines are a crude way of calculating the AB mag of the average flux.
    ##avg_wave = np.mean(ref_wave)
    ##mag_avg_flux = ABmag_nu(np.average(flux), avg_wave)   # see Bessell & Murphy 2012 eq 2.

    mag_single_V = ABmag_nu(flux_per_Hz[np.argmin(np.abs(SN_wave - V_wave))])


    if norm_meth == 'AVG':
        mag_zp = mag_avg_flux
    elif norm_meth == 'V_band':
        mag_zp = mag_single_V
    elif norm_meth == None:
        mag_zp = 0.

    mag_norm = -(mag - mag_zp)  # the minus sign is because we will plot E(V-X)

    ## calculate magnitude uncertainty
    ## note the extra factor of lambda*2/c gets canceled.
    if type(var)!=type(None):
        fr_err = np.sqrt(var)/flux
        mag_var = (FrFlx2mag*fr_err)**2


    if type(var)!=type(None):
        results = (mag_norm, mag_var, mag_single_V)
    else:
        results = (mag_norm, mag_single_V)
                              

    return results


def filter_features(features, wave):
        '''Returns a mask of boolean values the same size as
        the wave array.  True=wavelength not in features, False=wavelength
        is in features.
        
        Can be used like:
        
        mask = filter_features(FEAURES, wave)
        wave_no_features = wave[mask]
        flux_no_features = flux[mask]
        '''
        intersection = np.array([False]*wave.shape[0])
        for feature in features:
            intersection |= ((wave>feature[0])&(wave<feature[1]))
            
        return ~intersection
                
                
                
def log(msg=""):
        ## attach time stamp to print statements
        print "[{}] {}".format(strftime("%Y-%m-%d %H:%M:%S", localtime()), msg)



def grid_fit(phases, select_phases, pristine_11fe, obs_SN, N_BUCKETS = -1, u_guess=0., u_pad=0.15, u_steps=3, rv_guess=2.8, rv_pad=0.5, rv_steps=11, ebv_guess=1.0, ebv_pad=0.2, ebv_steps = 11, unfilt = False, norm_meth = 'AVG'):
    

        '''
            
        doctest implemented below.  May also want to look into nosetests.
        
        To run: 
        
        python mag_spectrum_fitting.py   
        
        # If I have these at the end of this function
        if __name__ == "__main__":
        import doctest
        doctest.testmod()
        
        
        # I don't like the previous approach, because the doctest is run everytime, and it wastes CPU time, so I have commented out those statement at the end
        of this fucntion.  Instead, do
        
        python -m doctest mag_spectrum_fitting.py -v
        
        
        ********    I should do this everytime before I commit.   ********
            
        The doctest below suppresses the print statements.
        
        doctest (NOTE: if the the length of the outputs don't match what's expected, python will complain result is not defined):
        
        (modified from:http://stackoverflow.com/questions/9949633/suppressing-print-as-stdout-python
        also take a look at this (the decorator method seems pretty elegant:
        http://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
        
        >>> obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
        >>> phases = [t[0] for t in obs_12cu]
        >>> pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
        >>> obs_SN = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.0, rv=2.8, loadmast=False, loadptf=False))
        >>> actualstdout = sys.stdout
        >>> sys.stdout = cStringIO.StringIO()
        >>> result = grid_fit(phases, pristine_11fe, obs_SN, u_steps = 5, rv_steps = 11, ebv_steps = 11)
        >>> sys.stdout = actualstdout
        >>> sys.stdout.write(str(np.round(result[0], decimals = 3)))
        [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        >>> sys.stdout.write(str(np.round(result[1], decimals = 3)))
        [ 2.8  2.8  2.8  2.8  2.8  2.8  2.8  2.8  2.8  2.8  2.8  2.8]
        >>> sys.stdout.write(str(np.round(result[2], decimals = 3)))
        [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
        '''
        
        if unfilt == True:
            FEATURES_ACTUAL = []
        else:
            FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),\
                              (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]

        
        
        
        
        tmpx = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, ebv_steps)
        tmpy = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, rv_steps)
        
        log( "SIZE OF GRID: {}".format(u_steps*rv_steps*ebv_steps) )
        log( "EBV SEARCH GRID:" )
        log( tmpx )
        log( "RV SEARCH GRID:" )
        log( tmpy )
        
        best_us = []
        best_rvs = []
        best_ebvs = []
        best_avs = []
        chi2dofs = []
        min_chi2s = []
        
        ebv_uncert_uppers = []
        ebv_uncert_lowers = []
        rv_uncert_uppers = []
        rv_uncert_lowers = []
        
        snake_hi_1sigs = []
        snake_lo_1sigs = []

        del_lamb = 1.
        band_steps = 1200
        V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)


        for phase_index, phase in zip(select_phases, [phases[select_phases]]):
            #        for phase_index in phases: # [0,]
            
            
                print '\n\n\n Phase_index', phase_index, '\n\n\n'
            
            #exit(1)
            
                ref = pristine_11fe[phase_index]
                ## Note: I have determined that ref_wave is equally spaced at 2A.
                ref_wave = ref[1].wave  # it is inefficient to define ref_wave in the for loop.  Should take it outside.  12/7/14.

                obs = obs_SN[phase_index]


                ## mask for spectral features not included in fit
                mask = filter_features(FEATURES_ACTUAL, ref_wave)

                print 'pristine_11fe type', type(pristine_11fe)
                print 'ref type', type(ref)
                print 'ref len', len(ref)
                print 'type ref[0]', type(ref[0])
                print 'type ref[1]', type(ref[1])
                print 'type ref[2]', type(ref[2])
                #exit(1)

                ref_mag_norm, return_wave, ref_mag_avg_flux, ref_V_mag, ref_mag_var, ref_calib_err, nanmask_ref, _ \
                            = extract_wave_flux_var(ref_wave, ref, N_BUCKETS = N_BUCKETS, mask = None, norm_meth = 'AVG')
                #exit(1)

                log()
                log( "Phase: {}".format(ref[0]) )
                
                ## 12cu or artificially reddened 11fe
                obs_mag_norm, _, obs_mag_avg_flux, obs_V_mag, obs_mag_var, obs_calib_err, nanmask_obs, obs_flux \
                            = extract_wave_flux_var(ref_wave, obs, N_BUCKETS = N_BUCKETS, mask = None, norm_meth = norm_meth)


                ## estimated of distance modulus
## ---> THIS CAN BE USED TO SHOW THAT V MAG DIFFERENCE IS A POOR INDICATOR FOR DISTANCE.  THIS MEANS TO USE V MAG NORMALIZATION TO TAKE OUT DISTANCE DIFFERENCE (AS FOLEY 2014 HAS DONE) IS NOT
## AN EFFECTIVE WAY. BUT THE MAG OF THE AVERAGE FLUX IS A CONSISTENT INDICATOR OF DISTANCE AND THAT'S WHY WE USE THIS AS THE NORMALIZATION TO TAKE OUT DISTANCE DIFFERENCE.
                del_mag_avg = obs_mag_avg_flux - ref_mag_avg_flux
                del_V_mag = obs_V_mag - ref_V_mag
                print '\n\n\n difference in magnitudes of average flux:', del_mag_avg
                print ' difference in V magnitudes:', del_V_mag, '\n\n\n'

                ## Total Variance.
                var = ref_mag_var + obs_mag_var  # NOT DEALING WITH BLOCKING FEATURES NOW 12/8/14
                #var = ref_mag_var[mask] + obs_mag_var[mask]
                
                ## hack thrown together to filter nan-values (which arrise from negative fluxes)
                ## find any rows with nan-values in C_inv matrix (there shouldn't be any)
                ## nanmask = np.array(~np.max(np.isnan(C_total_inv), axis=1))[:,0]
                ## merge mask with nan-masks from obs_interp_mag, and ref_mag (calc'd above)
                nanmask = nanmask_obs & nanmask_ref
                log( "num. points with negative flux discarded: {}".format(np.sum(~nanmask)) )
                
                ## remove nan values from var
                #var = var[nanmask]


                ## Determine whether 2D or 3D fit.
                if u_steps > 1:
                    ## 3D case
                    u = np.linspace(u_guess - u_pad, u_guess + u_pad, u_steps)
                    param_num = 3
                elif u_steps == 1:
                    ## 2D case: only use the central value of u
                    u = np.array([u_guess,])
                    param_num = 2



                ##  dof for CHI2
                dof = len(obs_mag_norm) - param_num
                ## blocked b/c I don't deal with any masks now  -- though the nanmask may be OK.     dof = np.sum(nanmask) - param_num  # (num. data points)-(num. parameters)
                log( "dof: {}".format(dof) )


                x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, ebv_steps)
                y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, rv_steps)

                CHI2 = np.zeros((len(u), len(x), len(y)))
                
                log( "Scanning CHI2 grid..." )
                for i, dist in enumerate(u):
                    for j, EBV in enumerate(x):
                        for k, RV in enumerate(y):
                                
                                ## unredden the reddened spectrum, convert to mag
                                if N_BUCKETS < 1:
                                    unred_mag_norm = extract_wave_flux_var(ref_wave, obs, N_BUCKETS = N_BUCKETS, mask = None, norm_meth = 'AVG', ebv = EBV, rv = RV)[0]
                                else:
                                    mag_avg_flux = unred_mag_norm = extract_wave_flux_var(ref_wave, obs, N_BUCKETS = N_BUCKETS, mask = None, norm_meth = 'AVG', ebv = EBV, rv = RV)[2]
                                    redden_fm(return_wave, SN_obs[1].flux, ebv, rv)




                                plt.figure()
                                plt.plot(return_wave, ref_mag_norm, 'k.')
                                plt.plot(return_wave, unred_mag_norm, 'r.')


                                
#                                unred_flux = redden_fm(ref_wave, obs_flux, EBV, RV)
#                                unred_flux_per_Hz = unred_flux * (ref_wave**2/c)
#                                unred_mag_avg_flux = ABmag_nu(np.mean(unred_flux_per_Hz))

#unred_mag_norm, unred_mag_single_V = flux2mag(unred_flux_per_Hz, unred_flux, ref_wave, unred_mag_avg_flux, norm_meth = norm_meth)
                                ## I should implement a better way to use mask -- right now, there is a lot of reptition that is unnecessary.
                              
                                
                                ## this is (unreddened 12cu mag - pristine 11fe mag)
                                ## blocked b/c I don't deal with masks now delta = unred_mag_norm[mask] - ref_mag_norm - dist # yes, unred_mag_norm and ref_mag_norm are treated slightly asym'ly -- something I
                                                                                   # should fix.  -XH



                                delta = unred_mag_norm - ref_mag_norm - dist # yes, unred_mag_norm and ref_mag_norm are treated slightly asym'ly -- something I
                                                                                   # should fix.  -XH
               
                                # convert to vector from array and filter nan-values
                                # blocked b/c I don't deal with mask now.  delta = delta[nanmask]
                                
                                ## Remember if I ever want to things the matrix way, one needs to converst an array to a matrix:
                                ##   delta_array = np.squeeze(np.asarray(delta))  # converting 1D matrix to 1D array.
                        
                               
                                CHI2[i, j, k] = np.sum(delta*delta/var)

#print 'RV, EBV, CHI2', RV, EBV, CHI2[i, j, k]



                CHI2_dof = CHI2/dof
                CHI2_dof_min = np.min(CHI2_dof)
                log( "min CHI2 per dof: {}".format(CHI2_dof_min) )
                delCHI2_dof = CHI2_dof - CHI2_dof_min

                plt.show()
                exit(1)

                ## plot power law reddening curve
                ##  pl_red_curve = redden_pl2(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
                ##  plt.plot(ref_wave_inv, pl_red_curve, 'r-')
                
                ##*********************************** find 1-sigma and 2-sigma uncertainties based on confidence **************************************

                ## report/save results

                
                mindex = np.where(delCHI2_dof == 0)   # Note argmin() only works well for 1D array.  -XH
                print 'mindex', mindex

                ## basically it's the two elements in mindex.  But each element is a one-element array; hence one needs an addition index of 0.
                mu, mx, my = mindex[0][0], mindex[1][0], mindex[2][0]
                print 'mindex', mindex
                print 'mu, mx, my', mu, mx, my
                best_u, best_rv, best_ebv = u[mu], y[my], x[mx]
                print 'best_u, best_rv, best_ebv', best_u, best_rv, best_ebv
                ## estimate of distance modulus
                best_av = best_rv*best_ebv

#print 'delCHI2_dof', delCHI2_dof


                best_fit_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
                snake_hi_1sig = deepcopy(best_fit_curve)
                snake_lo_1sig = deepcopy(best_fit_curve)

                maxebv_1sig, minebv_1sig = best_ebv, best_ebv
                maxrv_1sig, minrv_1sig = best_rv, best_rv
                for i, dist in enumerate(u):
                    for e, EBV in enumerate(x):
                        for r, RV in enumerate(y):
                            if delCHI2_dof[i, e, r] < 1.0:
                                maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                                minebv_1sig = np.minimum(minebv_1sig, EBV)
                                maxrv_1sig = np.maximum(maxrv_1sig, RV)
                                minrv_1sig = np.minimum(minrv_1sig, RV)

                                ## for plotting uncertainty snake.
                                ## the approach is trying to find the outmost contour of all excess curves with delta_chi2 < 1
                                ## As such it finds the region where the probability of a E(V-X) curve lies within which is 68%.
                                ## This typically is a (much) smaller region than the one enclosed between the F99 curve with (RV+sig_RV, EBV+sig_EBV)
                                ## and the one with (RV-sig_RV, EBV-sig_EBV).  I think it is the correct way of representing the 1-sigma uncertainty snake. -XH

                                ## AS reverses the order of j and k; I think I'm right.  In fact if the steps are different for RV and EBV
                                ## one gets an error message if j and k are reversed.
                                redden_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -EBV, RV, return_excess=True)
                                snake_hi_1sig = np.maximum(snake_hi_1sig, redden_curve)  # the result is the higher values from either array is picked.
                                snake_lo_1sig = np.minimum(snake_lo_1sig, redden_curve)


                print 'best_ebv, best_rv', best_ebv, best_rv
                print 'maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig', maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig
            
                
                ebv_uncert_upper = maxebv_1sig - best_ebv
                ebv_uncert_lower = best_ebv - minebv_1sig
                rv_uncert_upper = maxrv_1sig - best_rv
                rv_uncert_lower = best_rv - minrv_1sig
                
                print 'ebv_uncert_upper, ebv_uncert_lower', ebv_uncert_upper, ebv_uncert_lower
                print 'rv_uncert_upper, rv_uncert_lower', rv_uncert_upper, rv_uncert_upper
                
                print '\n\n\n rough estimate of distance modulus:', del_V_mag - best_av, '\n\n\n'

                log( "\t {}".format(mindex) )
                log( "\t u={} RV={} EBV={} AV={}".format(best_u, best_rv, best_ebv, best_av) )
                
                chi2dofs.append(CHI2_dof)
                #chi2_reductions.append(CHI2_dof)
                min_chi2s.append(CHI2_dof_min)
                best_us.append(best_u)
                best_rvs.append(best_rv)
                best_ebvs.append(best_ebv)
                best_avs.append(best_av)
                
                ebv_uncert_uppers.append(ebv_uncert_upper)
                ebv_uncert_lowers.append(ebv_uncert_lower)
                rv_uncert_uppers.append(rv_uncert_upper)
                rv_uncert_lowers.append(rv_uncert_lower)

                snake_lo_1sigs.append(snake_lo_1sig)
                snake_hi_1sigs.append(snake_hi_1sig)


        pprint( zip(phases, best_rvs, rv_uncert_uppers, rv_uncert_lowers, best_ebvs, ebv_uncert_uppers, ebv_uncert_lowers, best_avs, min_chi2s) )
                
        ## save results with date
        ##   filename = "spectra_mag_fit_results_{}.pkl".format(strftime("%H-%M-%S-%m-%d-%Y", gmtime()))


        if unfilt:
            filename = "spectra_mag_fit_results_UNFILTERED.pkl"
        else:
            filename = "spectra_mag_fit_results_FILTERED.pkl"


        cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                     'chi2dof': chi2dofs, 'u_steps': u_steps, 'rv_steps': rv_steps, 'ebv_steps': ebv_steps,
                     'u': u, 'x': x, 'y': y, 'ebv_uncert_upper': ebv_uncert_uppers, 'ebv_uncert_lower': ebv_uncert_lowers, \
                     'rv_uncert_upper': rv_uncert_uppers,'rv_uncert_lower': rv_uncert_lowers}, open(filename, 'wb'))
        
        log( "Results successfully saved in: {}".format(filename) )

        print 'in per_phase():', best_us, best_rvs, best_ebvs

        return snake_hi_1sigs, snake_lo_1sigs


def plot_photom_excess():
    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
    
    fig = plt.figure(figsize = (20, 12))
    
    
    plot_phase_excesses('SN2012CU', load_12cu_colors, RED_LAW, filters_bucket, zp_bucket)
    
    # format figure
    fig.suptitle('SN2012CU: Color Excess Per Phase', fontsize=TITLE_FONTSIZE)
    
    fig.text(0.5, .05, 'Inverse Wavelength ($1 / \mu m$)',
             fontsize=AXIS_LABEL_FONTSIZE, horizontalalignment='center')
        
    p1, = plt.plot(np.array([]), np.array([]), 'k--')
    p2, = plt.plot(np.array([]), np.array([]), 'r-')
    fig.legend([p1, p2], ['Fitzpatrick-Massa 1999*', 'Power-Law (Goobar 2008)'],
            loc=1, bbox_to_anchor=(0, 0, .97, .99), ncol=2, prop={'size':LEGEND_FONTSIZE})

    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.90, wspace=0.2, hspace=0.2)
    plt.show()





def plot_excess(phases, select_phases, title, info_dict, pristine_11fe, obs_SN, snake_hi_1sigs, snake_lo_1sigs):
    
    fig = plt.figure(figsize = (20, 12))
    
    #phases = [t[0] for t in obs_SN]
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    pmin, pmax = np.min(phases), np.max(phases)


    ref_wave = pristine_11fe[0][1].wave

    for i, phase_index, phase in izip(range(len(select_phases)), select_phases, [phases[select_phases]]):
        
        
        print "Plotting phase {} ...".format(phase)
        print 'i', i
        #exit(1)
        #        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        ax = plt.subplot(111)
        
        ref = pristine_11fe[phase_index]
        obs = obs_SN[phase_index]
        

        color_ref = extract_wave_flux_var(ref_wave, ref, norm_meth = 'V_band')[0]  #[0]: keep the 0th output.  Much more elegant than color_ref, _, _, _, _ = ...
        color_obs = extract_wave_flux_var(ref_wave, obs, norm_meth = 'V_band')[0]

        excess = color_obs - color_ref

### Keeping the next few lines for now since I may want to anchor the E(V-X) plot using a broadband V-mag.
#    V_band = [(5412., 5414., 'Vband')]
#    del_lamb = 1.
#    band_steps = 1200
#    V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)
#    print V_band_range
#    ref_V_mag = -2.5*np.log10(ref_interp(V_band_range).mean())  # need to add var's as weights.
#    obs_V_mag = -2.5*np.log10(obs_interp(V_band_range).mean())  # need to add var's as weights.


        # This way doesn't seem to give the correct V-band.  Hasn't looked into this too closely.  If the program works fine, can throw away these lines.  -XH, 11/30/14
        #  V_band = [(5300., 5500., 'Vband')]
        #  Vband_mask = filter_features(V_band, ref_wave) # Not the most efficient way of doing things, but this statement is here because ref_wave is inside the for loop -- also inefficient. Should fix this.
        #  ref_flux_V_mag = -2.5*np.log10(np.average(ref_flux[Vband_mask]))
        #  obs_flux_V_mag = -2.5*np.log10(np.average(obs_flux[Vband_mask]))
        

        best_ebv = info_dict['ebv'][i]
        best_rv  = info_dict['rv'][i]
        x = info_dict['x']
        y = info_dict['y']
        u = info_dict['u']
        
        
        ## convert effective wavelengths to inverse microns
        ref_wave_inv = 10000./ref_wave
        mfc_color = plt.cm.cool(5./11)
        
        ## plot excess (this is the data)
        plt.plot(ref_wave, excess, '.', color=mfc_color, ms=6, mec='none', mfc=mfc_color, alpha=0.8)



        ## plot best-fit reddening curve
        fm_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        plt.plot(ref_wave, fm_curve, 'k--')


        ## RV = 2.7, EBV = 1.02 reddenging curve.
        fm_curve27 = redden_fm(ref_wave, np.zeros(ref_wave.shape), -1.02*np.ones(best_ebv.shape), 2.7*np.ones(best_rv.shape), return_excess=True)
        plt.plot(ref_wave, fm_curve27, 'r-')

        ## Plot uncertainty snake.
        ax.fill_between(ref_wave, snake_lo_1sigs[i], snake_hi_1sigs[i], facecolor='black', alpha=0.3)

        ## plot where V band is.   -XH
        plt.plot([ref_wave.min(), ref_wave.max()], [0, 0] ,'--')
        plt.plot([V_wave, V_wave], [fm_curve.min(), fm_curve.max()] ,'--')
         



### FORMAT SUBPLOT ###

        ## print data on subplot

        ebv_uncert_upper = info_dict['ebv_uncert_upper'][i]
        ebv_uncert_lower = info_dict['ebv_uncert_lower'][i]
        rv_uncert_upper = info_dict['rv_uncert_upper'][i]
        rv_uncert_lower = info_dict['rv_uncert_lower'][i]



        ## Below essentially plots the region enclosed between the F99 curve with (RV+sig_RV, EBV+sig_EBV)
        ## and the one with (RV-sig_RV, EBV-sig_EBV).  I think this is the INCORRECT way of representing the 1-sigma uncertainty snake.
        ## See comments in the section where I compute 1-sigma uncertainties based on delta_chi2.
        ## Thus the following four statements are typically commented out.  Can remove them at some point  -XH, 11/30/2014
        ##        fm_curve_upper = redden_fm(ref_wave, np.zeros(ref_wave.shape), -(best_ebv + ebv_uncert_upper), best_rv + rv_uncert_upper, return_excess=True)
        ##        plt.plot(ref_wave_inv, fm_curve_upper, 'b-')
        ##
        ##        fm_curve_lower = redden_fm(ref_wave, np.zeros(ref_wave.shape), -(best_ebv - ebv_uncert_lower), best_rv - rv_uncert_lower, return_excess=True)
        ##        plt.plot(ref_wave_inv, fm_curve_lower, 'g-')




        plttext = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
        plttext = plttext.format(best_ebv, ebv_uncert_upper, ebv_uncert_lower,
                                 best_rv, rv_uncert_upper, rv_uncert_lower
                                 )
            
        ax.text(.95, .2, plttext, size=INPLOT_LEGEND_FONTSIZE,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax.transAxes)
         
        ## format subplot
        plt.xlim(3000, 10000)
        plt.ylim(-3.0, 2.0)

        if i%PLOTS_PER_ROW == 0:
            ax.set_title('Phase: {}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel('$E(V-X)$', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_title('{}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
            
            labels = ax.get_yticks().tolist()
            labels[0] = labels[-1] = ''
            ax.set_yticklabels(labels)

            labels = ax.get_xticks().tolist()
            labels[0] = labels[-1] = ''
            ax.set_xticklabels(labels)
                                                     
            plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
            plt.setp(ax.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)


        ## format figure
        fig.suptitle('{}: Color Excess'.format(title), fontsize=TITLE_FONTSIZE)
    
        fig.text(0.5,.05, 'Inverse Wavelength ($1 / \mu m$)',
             fontsize=AXIS_LABEL_FONTSIZE, horizontalalignment='center')
         
        p1, = plt.plot(np.array([]), np.array([]), 'k--')
        p2, = plt.plot(np.array([]), np.array([]), 'r-')
        fig.legend([p1, p2], ['Fitzpatrick-Massa 1999*', 'F99-RV27'],
                    loc=1, bbox_to_anchor=(0, 0, .97, .99), ncol=2, prop={'size':LEGEND_FONTSIZE})
         
        fig.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.90, wspace=0.2, hspace=0.2)
        filenm = filter(str.isalnum, title)+'.png' # to get rid of white space and parentheses; and add extension.
        fig.savefig(filenm)

##******************************************************** MAIN *******************************************************

if __name__=="__main__":
    '''
    
    Example:
    
    
    python mag_spectrum_fitting.py -obs_SN 'red_11fe' -N_bands -1 -select_phases 4 -u_guess 0. -u_pad 0.1 -u_steps 3 -rv_guess 2.8 -rv_pad 1.0 -rv_steps 11 -ebv_guess 1.0 -ebv_pad 0.2 -ebv_steps 11 -unfilt
    
    Note: it is possible to to add FEATURES_ACTUAL at the command line too.  It takes a little finessing.  When I'm ready to implement, try one or both of the following:
    One could use nargs = '*':
    http://stackoverflow.com/questions/10984769/argparse-optional-argument-with-list-tuple
    
    Or one can define one's own type:
    http://stackoverflow.com/questions/9978880/python-argument-parser-list-of-list-or-tuple-of-tuples
    
    '''
    

    ####### Converting inputs #######

    parser = argparse.ArgumentParser()
    parser.add_argument('-obs_SN', type = str)
    parser.add_argument('-select_phases',  '--select_phases', nargs='+', type=int)  # this can take a tuple: -select_phases 0 4  but the rest of the program can't handle more than
                                                                                    # one phases yet.  -XH 12/7/14
    parser.add_argument('-N_bands', type = int)
    parser.add_argument('-u_guess', type = float)
    parser.add_argument('-u_pad', type = float)
    parser.add_argument('-u_steps', type = int)
    parser.add_argument('-rv_guess', type = float)
    parser.add_argument('-rv_pad', type = float)
    parser.add_argument('-rv_steps', type = int)
    parser.add_argument('-ebv_guess', type = float)
    parser.add_argument('-ebv_pad', type = float)
    parser.add_argument('-ebv_steps', type = int)
    _ = parser.add_argument('-unfilt', '--unfilt', action='store_true')  # just another way to add an argument to the list.


    ### The '*' means there can be more than one positional arguments.
    #parser.add_argument('params', nargs = '*', type = float)
    args = parser.parse_args()
    print 'args', args
    
    obs_SN = args.obs_SN
    N_bands = args.N_bands
    u_guess = args.u_guess
    u_pad = args.u_pad
    u_steps = args.u_steps
    rv_guess = args.rv_guess
    rv_pad = args.rv_pad
    rv_steps = args.rv_steps
    ebv_guess = args.ebv_guess
    ebv_pad = args.ebv_pad
    ebv_steps = args.ebv_steps
    select_phases = np.array(args.select_phases) ## if there is only one phase select, it needs to be in the form of a 1-element array for all things to work.
    unfilt = args.unfilt
    
    ## load spectra, interpolate 11fe to 12cu phases (only first 12)
    obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in obs_12cu]
        
        
    pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    art_reddened_11fe = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.0, rv=2.8, del_mu=0.5, noiz=0.0, loadmast=False, loadptf=False))
    
        
    ## obs_SN is either an artificially reddened 11fe interpolated to the phases of 12cu, or 12cu itself.
    if obs_SN == 'red_11fe':
        obs_SN = art_reddened_11fe
    elif obs_SN == '12cu':
        obs_SN = obs_12cu
    
    #select_phases = np.array(select_phases)
#select_phases = np.array([0])

#print phases_to_fit
#exit(1)

    ## The following is a little hackish. i = 0 (unfilt = False) corresponds to the filtered case, and i = 1 (unfilt = True) corresponds to the unfiltered case.
    ## Also range(False) gives []; so I use range(unfilt + 1): range(False+1] gives [0], range[True+1] gives [0, 1].
    print 'unfilt', unfilt
    for i in [1]:  #range(unfilt+1):   right now, I only deal with the unfiltered case.  -XH 12/7/14
        snake_hi_1sigs, snake_lo_1sigs = grid_fit(phases, select_phases, pristine_11fe, obs_SN, N_BUCKETS = N_bands, u_guess=u_guess, u_pad=u_pad, u_steps=u_steps, rv_guess=rv_guess, rv_pad=rv_pad, rv_steps=rv_steps, ebv_guess=ebv_guess, ebv_pad=ebv_pad, ebv_steps=ebv_steps, norm_meth = 'AVG', unfilt = i)
        pkl_file = "spectra_mag_fit_results_FILTERED.pkl" if i == 0 else "spectra_mag_fit_results_UNFILTERED.pkl"
        info_dict = cPickle.load(open(pkl_file, 'rb'))
        save_name = "SN2012cu (Feature Filtered)" if i == 0 else "SN2012cu"
        plot_excess(phases, select_phases, save_name, info_dict, pristine_11fe, obs_SN, snake_hi_1sigs, snake_lo_1sigs)
    plt.show()
