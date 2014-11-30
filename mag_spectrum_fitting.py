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



# config
PLOTS_PER_ROW = 6
N_BUCKETS = 20
RED_LAW = redden_fm

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15

V_wave = 5413.5 
## ind_min = np.abs(red_curve).argmin()
## f99wv = np.array([ref_wave[ind_min-1],ref_wave[ind_min],ref_wave[ind_min+1]])
## f99ebv = np.array([red_curve[ind_min-1],red_curve[ind_min],red_curve[ind_min+1]])
## V_wave = np.interp(0., f99ebv, f99wv).  To figure out the wavelength of the "zero" of the F99 law.  Using different phases, the value varies a bit (my estimate: 5413.5+/-0.1).
## Haven't looked at this too closely.  But probably because of numerical error, such as during the interpolation process.  It's interesting
## that, reading through the fm_unred.py, which AS and ZR have adopted, it's not obvious what this value should be.
## Here I have arbitrarily chosen the first phase to infer what this wavelength should be.  -XH 11/18/14
## AS previously used V_wave = 5417.2


FrFlx2mag = 2.5/np.log(10)  #  =1.0857362

def ABmag(flux_per_Hz):
    return -2.5*np.log10(flux_per_Hz) - 48.6  ## Bessell & Murphy 2012.


def extract_wave_flux_var(ref_wave, SN, mask = None, norm_meth = 'AVG'):

    '''
    Added Nov 25, 2014.
    
    takes in 2 spectral pickle files from loader.py, extracts, interpolates and converts, and finally returns
    normalized magntiudes and magnitude variances.
    
    '''



    ## pristine_11fe
    #ref_wave = ref[1].wave


    SN_flux = SN[1].flux

    var = SN[1].error

    if (SN_flux <= 0).any():
        print "In extract_wave_flux_var():"
        print "some flux values are not positive:", SN_flux[np.where(SN_flux <= 0)]
        print "These values will be rejected below as nan for the log."
        print "(But it's better to deal with the non-pos values before taking the log (even before interpolation).  Something to deal with later.)"
        print "\n\n\n"



    flux_interp = interp1d(SN[1].wave, SN_flux)  # interp1d returns a function, which can be evaluated at any wavelength one would want.
                                                 # think of the two arrays supplied as the "training set".  So flux_interp() is a function.

    ## This is flux per frequency -- in order to calculate the AB magnitude -- see Bessell & Murphy eq 2 and eq A1; O'Donnell Astro 511 Lec 14.
    ## It doesn't seem to make any difference in terms of determining RV but it is the right way of doing things.  Magnitudes and colors calculated
    ## from these flux values should be directly comparable to AB mag's for broad bands, if that's what Andrew calculated for synthetic photometry.
    ## Does F99 assume a certain magnitude system?
    flux = flux_interp(ref_wave)#*(ref_wave**2)
    #flux_single_V = flux_interp(V_wave)#*(V_wave**2)

    var = interp1d(SN[1].wave, var)(ref_wave)


    # B-V color for 11fe
    #ref_B_V = -2.5*np.log10(ref_flux[np.abs(ref_wave - 4400).argmin()]/ref_flux[np.abs(ref_wave - 5413.5).argmin()])
    #print 'B-V for 11fe:', ref_B_V
                
    calib_err_mag = SN[2]

    ## convert flux, variance, and calibration error to magnitude space

#    mag_avg_flux = ABmag(np.average(flux))  # One shouldn't use the photon noise as the weight to find the average flux - see NB 11/22/14.
                                            # Note it's not the avg mag but the mag of the avg flux.
#    mag_single_V = ABmag(flux_single_V)

    mag_norm, mag_var, mag_avg_flux, mag_single_V = flux2mag(flux, ref_wave, var, norm_meth = norm_meth)
    
    
    if mask != None:
        mag_norm = mag_norm[mask]  # Note: mask has the same length as mag_norm, and contains a bunch of 0's and 1's (the 0's are where the blocked features are).
                                   # This is a very pythonic way of doing things: even though mask doesn't specifiy the indices of the wavelengths that should
                                   # be blocked, the operation mag_norm[mask] does just that.  One can think of mask as providing a truth table that tells python
                                   # which of the elements in mag_norm to keep and which to discard.  Yes, it doesn't make sense at first sight since mask doesn't
                                   # contain indices.  But it does work, and is the pythonic way!  -XH 11/25/14.


    # get mask for nan-values
    nanmask = ~np.isnan(mag_norm)
    

    return mag_norm, mag_avg_flux, mag_single_V, mag_var, calib_err_mag, nanmask, flux


def flux2mag(flux, ref_wave, var=None, norm_meth = 'AVG'):
    mag_var = None
    
    mag = ABmag(flux)
    mag_avg_flux = ABmag(np.average(flux))   # see Bessell & Murphy 2012 eq 2.
    mag_single_V = ABmag(flux[np.argmin(np.abs(ref_wave - V_wave))])


    if norm_meth == 'AVG':
        mag_zp = mag_avg_flux
    
    elif norm_meth == 'single_V':
        mag_zp = mag_single_V

    mag_norm = -(mag - mag_zp)  # the minus sign is because we will plot E(V-X)

    # calculate magnitude error
    if type(var)!=type(None):
        fr_err = np.sqrt(var)/flux
        mag_var = (FrFlx2mag*fr_err)**2


    if type(var)!=type(None):
        results = (mag_norm, mag_var, mag_avg_flux, mag_single_V)
    else:
        results = (mag_norm, mag_avg_flux, mag_single_V)
                              

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
        # attach time stamp to print statements
        print "[{}] {}".format(strftime("%Y-%m-%d %H:%M:%S", localtime()), msg)



def grid_fit(phases, pristine_11fe, obs_SN, u_guess=0., u_pad=0.15, u_steps=3, rv_guess=2.8, rv_pad=0.5, rv_steps=11, ebv_guess=1.0, ebv_pad=0.2, ebv_steps = 11):
    

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
        
        
        #FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
        #                      (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]


        # Use an empty list of features to fit for the entire spectrum:
        FEATURES_ACTUAL = []
        
        
        
        
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
        
        
        #V_band = [(5300., 5500., 'Vband')]

        del_lamb = 1.
        band_steps = 1200
        V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)


        for phase_index in xrange(len(phases)): # [0,]: # xrange((len(phases)):
            
            
                print '\n\n\n Phase_index', phase_index, '\n\n\n'
            
                ref = pristine_11fe[phase_index]
                ref_wave = ref[1].wave  ## I have determined that ref_wave is equally spaced at 2A.

#                ref_wave_R = ref_wave[1:]
#                ref_wave_L = ref_wave[:-1]
#                ref_wave_del = ref_wave_R - ref_wave_L
#                print ref_wave_R
#                print ref_wave_L
#                print np.max(ref_wave_del), np.min(ref_wave_del), np.mean(ref_wave_del)
#                plt.plot(ref_wave_L, ref_wave_del, 'k.')
#                #plt.ylim([-1., 5.])
#                plt.show()
#                exit(1)   ## This determined ref_wave are equally spaced at 2A.


                obs = obs_SN[phase_index]


                # mask for spectral features not included in fit
                mask = filter_features(FEATURES_ACTUAL, ref_wave)


                ref_mag_norm, ref_mag_avg_flux, ref_mag_single_V, ref_mag_var, ref_calib_err, nanmask_ref, _ = extract_wave_flux_var(ref_wave, ref, mask = mask, norm_meth = 'AVG')

#                        print 'ref_mag_norm', len(ref_mag_norm)
#                        print 'ref_wave', len(ref_wave)
#                        print 'mask', len(mask)
#                        
#                        exit(1)
#                        

                log()
                log( "Phase: {}".format(ref[0]) )
                


                ## 12cu/reddened 11fe

                obs_mag_norm, obs_mag_avg_flux, obs_mag_single_V, obs_mag_var, obs_calib_err, nanmask_obs, obs_flux = extract_wave_flux_var(ref_wave, obs, mask = mask, norm_meth = 'AVG')


                ## estimated of distance modulus
                del_mag_avg = obs_mag_avg_flux - ref_mag_avg_flux
                del_single_V_mag = obs_mag_single_V - ref_mag_single_V
                print '\n\n\n difference in magnitudes of average flux:', del_mag_avg
                print ' difference in single-wavelength V magnitudes:', del_single_V_mag, '\n\n\n'

                ## Total Variance.
                var = ref_mag_var[mask] + obs_mag_var[mask]
                
                
                
                #################################################
                # hack thrown together to filter nan-values (which arrise from negative fluxes)
                
                # find any rows with nan-values in C_inv matrix (there shouldn't be any)
                #nanmask = np.array(~np.max(np.isnan(C_total_inv), axis=1))[:,0]
                
                # merge mask with nan-masks from obs_interp_mag, and ref_mag (calc'd above)
                nanmask = nanmask_obs & nanmask_ref
                
                log( "num. points with negative flux discarded: {}".format(np.sum(~nanmask)) )
                
                # create temp version of C_total_inv without rows/columns corresponding to nan-values
                var = var[nanmask]


                ## Determine whether 2D or 3D fit.
                if u_steps > 1:
                    u = np.linspace(u_guess - u_pad, u_guess + u_pad, u_steps)
                    param_num = 3
                elif u_steps == 1:
                    u = np.array([u_guess,])
                    param_num = 2


                #################################################
                
                ## for calculation of CHI2 per dof
                dof = np.sum(nanmask) - param_num  # (num. data points)-(num. parameters)
                log( "dof: {}".format(dof) )
                
                #################################################
                


                x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, ebv_steps)
                y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, rv_steps)
                
                #X, Y = np.meshgrid(x, y)
                CHI2 = np.zeros((len(u), len(x), len(y)))
                
                log( "Scanning CHI2 grid..." )
                for i, dist in enumerate(u):
                    for j, EBV in enumerate(x):
                        for k, RV in enumerate(y):
                                
                                # unredden the reddened spectrum, convert to mag
                                unred_flux = redden_fm(ref_wave, obs_flux, EBV, RV)
                                unred_mag_norm, unred_mag_avg_flux, unred_mag_single_V = flux2mag(unred_flux, ref_wave, norm_meth = 'AVG')
                                ## I should implement a better way to use mask -- right now, there is a lot of reptition that is unnecessary.
                              
                                
                                # this is (unreddened 12cu mag - pristine 11fe mag)
                                delta = unred_mag_norm[mask] - ref_mag_norm - dist # yes, unred_mag_norm and ref_mag_norm are treated slightly asym'ly -- something I
                                                                                   # should fix.  -XH
                                
                                # convert to vector from array and filter nan-values
                                delta = delta[nanmask]
                                
                                #delta_array = np.squeeze(np.asarray(delta))  # converting 1D matrix to 1D array.
                                ## ----->I shoudl fix ylim<-------------------
                                #tmp_wave = ref_wave[mask]
                                #fig = plt.figure()
                                #plt.plot(tmp_wave[nanmask], delta_array, 'ro')
                               
                               
                                CHI2[i, j, k] = np.sum(delta*delta/var)


                CHI2_dof = CHI2/dof
                CHI2_dof_min = np.min(CHI2_dof)
                log( "min CHI2 per dof: {}".format(CHI2_dof_min) )
                delCHI2_dof = CHI2_dof - CHI2_dof_min
                
                
                # plot power law reddening curve
                #pl_red_curve = redden_pl2(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
                #plt.plot(ref_wave_inv, pl_red_curve, 'r-')
                
                ##*********************************** find 1-sigma and 2-sigma errors based on confidence **************************************

                ### report/save results
                
                #print 'CHI2', CHI2
                
                mindex = np.where(delCHI2_dof == 0)   # Note argmin() only works well for 1D array.  -XH
                print 'mindex', mindex

                # basically it's the two elements in mindex.  But each element is a one-element array; hence one needs an addition index of 0.
                mu, mx, my = mindex[0][0], mindex[1][0], mindex[2][0]
                print 'mindex', mindex
                print 'mu, mx, my', mu, mx, my
                best_u, best_rv, best_ebv = u[mu], y[my], x[mx]
                print 'best_u, best_rv, best_ebv', best_u, best_rv, best_ebv
                ## estimate of distance modulus
                best_av = best_rv*best_ebv

                print 'delCHI2_dof', delCHI2_dof


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

                
                
                
                print '\n\n\n rough estimate of distance modulus:', del_single_V_mag - best_av, '\n\n\n'

               
               
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


#        return best_rvs, best_ebvs


        pprint( zip(phases, best_rvs, rv_uncert_uppers, rv_uncert_lowers, best_ebvs, ebv_uncert_uppers, ebv_uncert_lowers, best_avs, min_chi2s) )
                
        ## save results with date
        #                filename = "spectra_mag_fit_results_{}.pkl".format(strftime("%H-%M-%S-%m-%d-%Y", gmtime()))

        filename = "spectra_mag_fit_results_FILTERED.pkl"
#        cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
#                        'chi2': chi2s, 'chi2_reductions': chi2_reductions, 'u_steps': u_steps, 'rv_steps': rv_steps, 'ebv_steps': ebv_steps,
#                        'u': u, 'x': x, 'y': y, 'X': X, 'Y': Y},
#                        open(filename, 'wb'))

        cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                     'chi2dof': chi2dofs, 'u_steps': u_steps, 'rv_steps': rv_steps, 'ebv_steps': ebv_steps,
                     'u': u, 'x': x, 'y': y, 'ebv_uncert_upper': ebv_uncert_uppers, 'ebv_uncert_lower': ebv_uncert_lowers, \
                     'rv_uncert_upper': rv_uncert_uppers,'rv_uncert_lower': rv_uncert_lowers}, open(filename, 'wb'))
        
        log( "Results successfully saved in: {}".format(filename) )



#best_rvs, best_ebvs = perphase_fit()
        print 'in per_phase():', best_us, best_rvs, best_ebvs
        #        print 'in per_phase():', type(best_rvs), type(best_ebvs)

#exit(1)
        return snake_hi_1sigs, snake_lo_1sigs



def plot_excess(title, info_dict, pristine_11fe, obs_SN, snake_hi_1sigs, snake_lo_1sigs):
    
    fig = plt.figure(figsize = (20, 12))
    
    #obs_SN = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in obs_SN]
    
    #pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    pmin, pmax = np.min(phases), np.max(phases)
    
#    best_ebv = info_dict['ebv'][0]
#    best_rv  = info_dict['rv'][0]


    ref_wave = pristine_11fe[0][1].wave
   
   
#exit(1)

    for i, phase in enumerate(phases):
        
        
        print "Plotting phase {} ...".format(phase)
        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        
        ref = pristine_11fe[i]
        obs = obs_SN[i]
        

#ref_wave = ref[1].wave
        

        color_ref = extract_wave_flux_var(ref_wave, ref, norm_meth = 'single_V')[0]  #[0]: keep the 0th output.  Much more elegant than color_ref, _, _, _, _ = ...
        color_obs = extract_wave_flux_var(ref_wave, obs, norm_meth = 'single_V')[0]

        excess = color_obs - color_ref

### Keeping the next few lines for now since I may want to anchor the E(V-X) plot using a broadband V-mag.

#    #V_band = [(5412., 5414., 'Vband')]
# del_lamb = 1.
#    band_steps = 1200
#    V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)
#    print V_band_range


        #Vband_mask = filter_features(V_band, ref_wave) # Not the most efficient way of doing things, but this statement is here because ref_wave is inside the for loop -- also inefficient. Should fix this.

#ref_V_mag = -2.5*np.log10(ref_interp(V_band_range).mean())  # need to add var's as weights.
#obs_V_mag = -2.5*np.log10(obs_interp(V_band_range).mean())  # need to add var's as weights.
        
        # This way seems to give wrong answer.
        #          ref_flux_V_mag = -2.5*np.log10(np.average(ref_flux[Vband_mask]))
        #        obs_flux_V_mag = -2.5*np.log10(np.average(obs_flux[Vband_mask]))
        
        
        
        
        # convert effective wavelengths to inverse microns
        



        best_ebv = info_dict['ebv'][i]
        best_rv  = info_dict['rv'][i]
        x = info_dict['x']
        y = info_dict['y']
        u = info_dict['u']
        #        print len(chi2dof)
# print chi2dof[0].shape

        ref_wave_inv = 10000./ref_wave
        mfc_color = plt.cm.cool(5./11)
        
        ## plot excess (this is the data)
        plt.plot(ref_wave_inv, excess, '.', color=mfc_color, ms=6, mec='none', mfc=mfc_color, alpha=0.8)



        ## plot best-fit reddening curve
        fm_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        plt.plot(ref_wave_inv, fm_curve, 'k--')


        ## RV = 2.7, EBV = 1.02 reddenging curve.
        fm_curve27 = redden_fm(ref_wave, np.zeros(ref_wave.shape), -1.02*np.ones(best_ebv.shape), 2.7*np.ones(best_rv.shape), return_excess=True)
        plt.plot(ref_wave_inv, fm_curve27, 'r-')



        ## plot error snake
#        x = info_dict['x']
#        y = info_dict['y']
#        chi2dof = info_dict['chi2dof'][i]


        ax.fill_between(10000./ref_wave, snake_lo_1sigs[i], snake_hi_1sigs[i], facecolor='black', alpha=0.3)



        ## plot where V band is.   -XH
        plt.plot([ref_wave_inv.min(), ref_wave_inv.max()], [0, 0] ,'--')
        plt.plot([1e4/V_wave, 1e4/V_wave], [fm_curve.min(), fm_curve.max()] ,'--')
         



### FORMAT SUBPLOT ###

        ## print data on subplot

        ebv_uncert_upper = info_dict['ebv_uncert_upper'][i]
        ebv_uncert_lower = info_dict['ebv_uncert_lower'][i]
        rv_uncert_upper = info_dict['rv_uncert_upper'][i]
        rv_uncert_lower = info_dict['rv_uncert_lower'][i]



        ## Below essentially plots the region enclosed between the F99 curve with (RV+sig_RV, EBV+sig_EBV)
        ## and the one with (RV-sig_RV, EBV-sig_EBV).  I think this is the INCORRECT way of representing the 1-sigma uncertainty snake.
        ##  See comments in plot_snake(). Thus the following four statements are typically commented out.  -XH
        ##        fm_curve_upper = redden_fm(ref_wave, np.zeros(ref_wave.shape), -(best_ebv + ebv_uncert_upper), best_rv + rv_uncert_upper, return_excess=True)
        ##        plt.plot(ref_wave_inv, fm_curve_upper, 'b-')
        ##
        ##        fm_curve_lower = redden_fm(ref_wave, np.zeros(ref_wave.shape), -(best_ebv - ebv_uncert_lower), best_rv - rv_uncert_lower, return_excess=True)
        ##        plt.plot(ref_wave_inv, fm_curve_lower, 'g-')




        plttext = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
        plttext = plttext.format(best_ebv, ebv_uncert_upper, ebv_uncert_lower,
                                 best_rv, rv_uncert_upper, rv_uncert_lower
                                 )
            
        ax.text(.95, .98, plttext, size=INPLOT_LEGEND_FONTSIZE,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax.transAxes)
         
        ## format subplot
        plt.xlim(1.0, 3.0)
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

#plt.show()



if __name__=="__main__":
    
    

    ####### Converting inputs #######

    parser = argparse.ArgumentParser()
    parser.add_argument('-obs_SN', type = str)
#    parser.add_argument('-Gn', '--Gn', action='store_true')
#    parser.add_argument('-LN', '--LN', action='store_true')
#    parser.add_argument('-plot_dmap', '--plot_dmap', action='store_true')
#    parser.add_argument('-save_dmap_data', '--save_dmap_data', action='store_true')
#    parser.add_argument('-run_ini_ind', type = int)
#    parser.add_argument('-use_saved', '--use_saved', action='store_true')
#    parser.add_argument('-disks', '--disks',  action='store_true')
#    parser.add_argument('-disksAv', '--disksAv',  action='store_true')
#    parser.add_argument('-Rnoiz', '--Rnoiz',  action='store_true')
#    parser.add_argument('-file_loc', type = str)
#    _ = parser.add_argument('-Rv', '--calcRv', action='store_true')  # just another way to add an argument to the list.

    ### The '*' means there can be more than one positional arguments.
    #parser.add_argument('params', nargs = '*', type = float)
    args = parser.parse_args()
    print 'args', args
    obs_SN = args.obs_SN
#    Ndim = args.Ndim
#    AvScale = args.AvScale
#    AvMean = args.AvMean
#    AvLNScale = args.AvLNScale
#    AvLNMin = args.AvLNMin
#    t_start = args.t_start
#    t_end = args.t_end
#    tsteps = args.tsteps
#    ntrials = args.ntrials
#    Gn = args.Gn
#    LN = args.LN
#    plot_dmap = args.plot_dmap
#    save_dmap_data = args.save_dmap_data
#    run_ini_ind = args.run_ini_ind
#    use_saved = args.use_saved
#    day_fixed1 = args.day_fixed1
#    day_fixed2 = args.day_fixed2
#    disks = args.disks
#    disksAv = args.disksAv
#    Rnoiz = args.Rnoiz
#    calcRv = args.calcRv
#    file_loc = args.file_loc

    
    
    ## load spectra, interpolate 11fe to 12cu phases (only first 12)
    obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in obs_12cu]
        
        
    pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    art_reddened_11fe = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.0, rv=2.8, del_mu=0.5, noiz=0.0, loadmast=False, loadptf=False))
    
        
    ########################
    # Choose 'SNobs' to be either an artificially reddened 11fe interpolated
    # to the phases of 12cu, or just choose 12cu itself.
    #
    if obs_SN == 'red_11fe':
        obs_SN = art_reddened_11fe
    elif obs_SN == '12cu':
        obs_SN = obs_12cu
    #
    ########################


    snake_hi_1sigs, snake_lo_1sigs = grid_fit(phases, pristine_11fe, obs_SN, u_guess=0., u_pad=0.1, u_steps = 3, rv_guess=2.8, rv_pad=1., rv_steps=41, ebv_guess=1.0, ebv_pad=0.2, ebv_steps = 51)
    info_dict1 = cPickle.load(open("spectra_mag_fit_results_FILTERED.pkl", 'rb'))
    info_dict2 = cPickle.load(open("spectra_mag_fit_results_UNFILTERED.pkl", 'rb'))
                
    i = 0
    for t in zip(["SN2012cu (Feature Filtered)", "SN2012cu"], [info_dict1, info_dict2], pristine_11fe, obs_SN):
        if i > 0: break   # this is to not plot the unblocked fit.
        plot_excess(t[0], t[1], pristine_11fe, obs_SN, snake_hi_1sigs, snake_lo_1sigs)
        i += 1

