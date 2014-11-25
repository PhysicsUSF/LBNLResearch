'''
::AUTHOR::
Andrew Stocker


::Modified by:
Xiaosheng Huang
    
    
'''

#from __future__ import print_function


import sys
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


def extract_wave_flux_var(ref_wave, SN, mask):

    '''
    Added Nov 25, 2014.
    
    takes in 2 spectral pickle files from loader.py, extracts, interpolates and converts, and finally returns
    normalized magntiudes and magnitude variances.
    
    '''



    ## pristine_11fe
    #ref_wave = ref[1].wave

    SN_flux = SN[1].flux

    if (SN_flux <= 0).any():
        print "In extract_wave_flux_var():"
        print "some flux values are not positive:", SN_flux[np.where(SN_flux <= 0)]
        print "These values will be rejected below as nan for the log."
        print "(But it's better to deal with the non-pos values before taking the log (even before interpolation).  Something to deal with later.)"
        print "\n\n\n"



    flux = interp1d(SN[1].wave, SN_flux)(ref_wave)



    # B-V color for 11fe
    #ref_B_V = -2.5*np.log10(ref_flux[np.abs(ref_wave - 4400).argmin()]/ref_flux[np.abs(ref_wave - 5413.5).argmin()])
    #print 'B-V for 11fe:', ref_B_V
                
    var = SN[1].error
    calib_err = SN[2]


#    ref_interp = interp1d(ref_wave, ref_flux)   # What does this accomplish?  -XH 11/25/14
    
    ## convert flux, variance, and calibration error to magnitude space
    
    mag_norm, mag_var, calib_err_mag \
    = flux2mag(flux, var, calib_err)
    
    # normalize for later use
    #Vband_mask = filter_features(V_band, ref_wave) # Not the most efficient way of doing things, but this statement is here because ref_wave is inside the for loop -- also inefficient. Should fix this.
    #ref_V_mag = -2.5*np.log10(ref_interp(V_band_range).mean())
    
    #mag_norm = mag - flux_avg_mag
    
    
    # get mask for nan-values
    nanmask = ~np.isnan(mag_norm[mask])
    

    return mag_norm, mag_var, calib_err, nanmask, flux

def flux2mag(flux, var=None, calibration_err=None):
    mag_var = None
    calibration_err_mag = None
    
    mag = -2.5*np.log10(flux)
    flux_avg_mag = -2.5*np.log10(np.average(flux))  # One shouldn't use the photon noise as the weight to find the average flux - see NB 11/22/14.
    # the _flux_ in the name is to emphasize: it's not the avg mag but the mag of the avg flux.
    mag_norm = mag - flux_avg_mag

    # calculate magnitude error
    if type(var)!=type(None):
        fr_err = np.sqrt(var)/flux
        mag_var = (1.0857362*fr_err)**2  # 2.5/np.log(10) = 1.0857362

    # calculate calibration error in mag space
    if type(calibration_err)!=type(None):
        calibration_err_mag = 1.0857362*calibration_err
    
    results = tuple([r for r in (mag_norm, mag_var, calibration_err_mag) if type(r)!=type(None)])

    return (results, results[0])[len(results)==1]


def plot_snake(ax, rng, init, red_law, x, y, CHI2, plot2sig=False):
    snake_hi_1sig = deepcopy(init)
    snake_lo_1sig = deepcopy(init)
    if plot2sig:
        snake_hi_2sig = deepcopy(init)
        snake_lo_2sig = deepcopy(init)
    
    for i, EBV in enumerate(x):
        for j, RV in enumerate(y):
            _chi2 = CHI2[j,i]
            if _chi2<1.00:
                red_curve = red_law(rng, np.zeros(rng.shape), -EBV, RV, return_excess=True)
                ind_min = np.abs(red_curve).argmin()
                snake_hi_1sig = np.maximum(snake_hi_1sig, red_curve)
                snake_lo_1sig = np.minimum(snake_lo_1sig, red_curve)
            elif plot2sig and _chi2<4.00:
                red_curve = red_law(rng, np.zeros(rng.shape), -EBV, RV, return_excess=True)
                snake_hi_2sig = np.maximum(snake_hi_2sig, red_curve)
                snake_lo_2sig = np.minimum(snake_lo_2sig, red_curve)
    
    ax.fill_between(10000./rng, snake_lo_1sig, snake_hi_1sig, facecolor='black', alpha=0.3)
    if plot2sig:
        ax.fill_between(10000./rng, snake_lo_2sig, snake_hi_2sig, facecolor='black', alpha=0.1)
    
    return interp1d(rng, snake_lo_1sig), interp1d(rng, snake_hi_1sig)


## This function really should be imported from mag_spectrum_fitting.py if I can get that to work.
#def filter_features(features, wave):
#    intersection = np.array([False]*wave.shape[0])
#    for feature in features:
#        intersection |= ((wave>feature[0])&(wave<feature[1]))
#    
#    return ~intersection



def grid_fit(phases, pristine_11fe, obs_SN, rv_guess = 2.7, rv_pad = 0.5, ebv_guess = 1.0, ebv_pad = 0.2, steps = 11):
    

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
        >>> art_reddened_11fe = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.0, rv=2.7, loadmast=False, loadptf=False))
        >>> actualstdout = sys.stdout
        >>> sys.stdout = cStringIO.StringIO()
        >>> result = grid_fit(phases, pristine_11fe, art_reddened_11fe, rv_guess = 2.7, rv_pad = 0.5, ebv_guess = 1.0, ebv_pad = 0.2, steps = 11)
        >>> sys.stdout = actualstdout
        >>> sys.stdout.write(str(np.round(result[0], decimals = 3)))
        [ 2.7  2.7  2.7  2.7  2.7  2.7  2.7  2.7  2.7  2.7  2.7  2.7]
        >>> sys.stdout.write(str(np.round(result[1], decimals = 3)))
        [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
        '''


        # config
#        ebv_guess = 1.0
#        ebv_pad = 0.2
#        
#        rv_guess = 2.7
#        rv_pad = 0.5
#        
#        steps = 11
        #steps = 120
        
        
        FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
                               (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]


        # Use an empty list of features to fit for the entire spectrum:
        #FEATURES_ACTUAL = []
        
        
#        f1 = open('workfile', 'w')
#        f1.write('This is a test')
#        #print f, 'This is a test'
#        f1.close()
#        exit(1)

        ########################
        ### helper functions ###


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
        
        
        
        
        ########################
        
        
        
        def perphase_fit():
                
                tmpx = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                tmpy = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                
                log( "SIZE OF GRID: {}".format(steps) )
                log( "EBV SEARCH GRID:" )
                log( tmpx )
                log( "RV SEARCH GRID:" )
                log( tmpy )
                
                best_rvs = []
                best_ebvs = []
                best_avs = []
                chi2s = []
                min_chi2s = []
                chi2_reductions = []
                
                #V_band = [(5300., 5500., 'Vband')]

                del_lamb = 1.
                V_wave = 5413.5  # see my mag_spectrum_plot_excess.py
                band_steps = 1200
                V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)


                for phase_index in xrange(len(phases)): # [0,]: # xrange((len(phases)):
                    
                    
                        print '\n\n\n Phase_index', phase_index, '\n\n\n'
                    
                        ref = pristine_11fe[phase_index]
                        ref_wave = ref[1].wave

                        obs = obs_SN[phase_index]

                        # mask for spectral features not included in fit
                        mask = filter_features(FEATURES_ACTUAL, ref_wave)

                        ref_mag_norm, ref_mag_var, ref_calib_err, nanmask_ref, _ = extract_wave_flux_var(ref_wave, ref, mask)

                        log()
                        log( "Phase: {}".format(ref[0]) )
                        
    
                        # get mask for nan-values
                        #nanmask_ref = ~np.isnan(ref_mag_norm[mask])
                        
                        # 12cu/reddened 11fe

                        obs_mag_norm, obs_mag_var, obs_calib_err, nanmask_obs, obs_flux = extract_wave_flux_var(ref_wave, obs, mask)


                        # Total Variance.
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

                        #################################################
                        
                        
                        
                        # for calculation of CHI2 per dof
                        CHI2_reduction = np.sum(nanmask) - 2  # (num. data points)-(num. parameters)
                        log( "CHI2 reduction: {}".format(CHI2_reduction) )
                        
                        #################################################
                        
                        x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                        y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                        
                        X, Y = np.meshgrid(x, y)
                        CHI2 = np.zeros( X.shape )
                        
                        log( "Scanning CHI2 grid..." )
                        for j, EBV in enumerate(x):
                                for k, RV in enumerate(y):
                                        
                                        # unredden the reddened spectrum, convert to mag
                                        unred_flux = redden_fm(ref_wave, obs_flux, EBV, RV)
                                        unred_mag_norm = flux2mag(unred_flux)
                                        
                                        unred_interp = interp1d(ref_wave, unred_flux)  # what's purpose of this?  also what is ref_wave used here,
                                                                                       # instead of obs_wave, as is done in excess_plot().
                                                                                       # ----->  Need to truly understand how interp1d works. <--------

                                        # this is unreddened 12cu mag - pristine 11fe mag
                                        delta = unred_mag_norm[mask]-ref_mag_norm[mask]
                                        tmp_wave = ref_wave[mask]
                                        # convert to vector from array and filter nan-values
                                        delta = delta[nanmask]
                                        
                                        #delta_array = np.squeeze(np.asarray(delta))  # converting 1D matrix to 1D array.
                                        ## ----->I shoudl fix ylim<-------------------
                                        #fig = plt.figure()
                                        #plt.plot(tmp_wave[nanmask], delta_array, 'ro')
                                       
                                        
                                        # The original equation is delta.T * C_inv * delta, but delta
                                        # is already a row vector in numpy so it is the other way around.
                                        CHI2[k,j] = np.sum(delta*delta/var)




                        ### report/save results
                        
                        chi2_min = np.min(CHI2)
                        log( "min chi2: {}".format(chi2_min) )
                        
                        mindex = np.where(CHI2==chi2_min)
                        mx, my = mindex[1][0], mindex[0][0]
                        
                        log( "\t {}".format(mindex) )
                        log( "\t RV={} EBV={} AV={}".format(y[my], x[mx], x[mx]*y[my]) )
                        
                        chi2s.append(CHI2)
                        chi2_reductions.append(CHI2_reduction)
                        min_chi2s.append(chi2_min)
                        best_rvs.append(y[my])
                        best_ebvs.append(x[mx])
                        best_avs.append(x[mx]*y[my])

                return best_rvs, best_ebvs

#plt.show()
                pprint( zip(phases, best_rvs, best_ebvs, best_avs, min_chi2s) )
                
                # save results
                #                filename = "spectra_mag_fit_results_{}.pkl".format(strftime("%H-%M-%S-%m-%d-%Y", gmtime()))

                filename = "spectra_mag_fit_results_FILTERED.pkl"
                cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                                'chi2': chi2s, 'chi2_reductions': chi2_reductions, 'steps': steps,
                                'x': x, 'y': y, 'X': X, 'Y': Y},
                                open(filename, 'wb'))
                
                log( "Results successfully saved in: {}".format(filename) )



        best_rvs, best_ebvs = perphase_fit()
        print 'in per_phase():', best_rvs, best_ebvs
        print 'in per_phase():', type(best_rvs), type(best_ebvs)

        return best_rvs, best_ebvs

#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()



def plot_excess(title, info_dict, pristine_11fe, obs_SN):
    
    fig = plt.figure(figsize = (20, 12))#, dpi = 10)
    
    #obs_SN = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in obs_SN]
    
    #pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    pmin, pmax = np.min(phases), np.max(phases)
    
    
    ## To figure out the wavelength of the "zero" of the F99 law.  Using different phases, the value varies a bit (my estimate: 5413.5+/-0.1).
    ## Haven't looked at this too closely.  But probably because of numerical error, such as during the interpolation process.  It's interesting
    ## that, reading through the fm_unred.py, which AS and ZR have adopted, it's not obvious what this value should be.
    ## Here I have arbitrarily chosen the first phase to infer what this wavelength should be.  -XH 11/18/14
    best_ebv = info_dict['ebv'][0]
    best_rv  = info_dict['rv'][0]
    ref_wave = pristine_11fe[0][1].wave
    red_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
    ind_min = np.abs(red_curve).argmin()
    f99wv = np.array([ref_wave[ind_min-1],ref_wave[ind_min],ref_wave[ind_min+1]])
    f99ebv = np.array([red_curve[ind_min-1],red_curve[ind_min],red_curve[ind_min+1]])
    V_wave = np.interp(0., f99ebv, f99wv)
    print 'zero EBV wavelength (V band wavelength):', V_wave  # As I said, this value should be close to 5413.5 A.
    
    
    V_wave = 5413.5  # See the comment for the above block of code.
    # AS previously used V_wave = 5417.2
    
    
    V_band = [(5412., 5414., 'Vband')]
    del_lamb = 1.
    band_steps = 1200
    V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)
    print V_band_range
    
    
    for i, phase in enumerate(phases):
        
        
        print "Plotting phase {} ...".format(phase)
        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        
        ref = pristine_11fe[i]
        obs = obs_SN[i]
        
        best_ebv = info_dict['ebv'][i]
        best_rv  = info_dict['rv'][i]
        
        ref_wave = ref[1].wave
        ref_flux = ref[1].flux
        ref_var = ref[1].error    # 11fe variance.
        
        ref_interp = interp1d(ref_wave, ref_flux)   # why is this step necessary?? Maybe to get single-lambda V-band mag. -XH, 11/25/14
        obs_interp = interp1d(obs[1].wave, obs[1].flux)

        obs_flux = obs_interp(ref_wave)
        
        
        #obs_flux = obs_interp  # the type of obs_interp is <class 'scipy.interpolate.interpolate.interp1d'>, and not just an array.  It probably behaves as a function. One could also do obs_interp_flux = interp1d(obs_wave, obs[1].flux)(ref_wave) -XH Nov 18, 2014
        obs_var  = interp1d(obs[1].wave, obs[1].error)(ref_wave)  # 12cu variance.
        
        #Vband_mask = filter_features(V_band, ref_wave) # Not the most efficient way of doing things, but this statement is here because ref_wave is inside the for loop -- also inefficient. Should fix this.
        
        ## single wavelength magnitude
        ref_single_wave_mag = (-2.5*np.log10(ref_flux))
        obs_single_wave_mag = (-2.5*np.log10(obs_flux))
        
        
        #excess_ref = (-2.5*np.log10(np.mean(ref_flux))) - (-2.5*np.log10(ref_flux))  # need to add var's as weights.
        #excess_obs = (-2.5*np.log10(np.mean(obs_flux))) - (-2.5*np.log10(obs_flux))  # need to add var's as weights.
        
        
        
        ref_single_V_mag = -2.5*np.log10(ref_interp(V_wave))
        obs_single_V_mag = -2.5*np.log10(obs_interp(V_wave))
        
        ref_V_mag = -2.5*np.log10(ref_interp(V_band_range).mean())  # need to add var's as weights.
        obs_V_mag = -2.5*np.log10(obs_interp(V_band_range).mean())  # need to add var's as weights.
        
        # This way seems to give wrong answer.
        #          ref_flux_V_mag = -2.5*np.log10(np.average(ref_flux[Vband_mask]))
        #        obs_flux_V_mag = -2.5*np.log10(np.average(obs_flux[Vband_mask]))
        
        #        color_ref =  ref_single_V_mag - ref_single_wave_mag
        #        color_obs =  obs_single_V_mag - obs_single_wave_mag
        
        color_ref = ref_V_mag - ref_single_wave_mag
        color_obs = obs_V_mag - obs_single_wave_mag
        
        
        print '\n\n\n'
        print 'single lambda V band for 11fe', ref_single_V_mag
        print 'V band for 11fe', ref_V_mag
        print 'single lambda V band for 12cu', obs_single_V_mag
        print 'V band for 12cu', obs_V_mag
        
        print 'ABS(Avg_mag - V_mag for 11fe) - (Avg_mag - V_mag for 12cu)', np.abs(-2.5*np.log10(ref_interp(V_wave)/np.mean(ref_flux)) - -2.5*np.log10(obs_interp(V_wave)/np.mean(obs_flux)))
        
        
        print '\n\n\n'
        
        
        excess = color_obs - color_ref
        
        
        # convert effective wavelengths to inverse microns
        ref_wave_inv = 10000./ref_wave
        mfc_color = plt.cm.cool(5./11)
        
        # plot excess
        plt.plot(ref_wave_inv, excess, '.', color=mfc_color, ms=6, mec='none', mfc=mfc_color, alpha=0.8)
            
        # plot reddening curve
        fm_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        plt.plot(ref_wave_inv, fm_curve, 'k--')
         
        # plot where V band is.
        plt.plot([ref_wave_inv.min(), ref_wave_inv.max()], [0, 0] ,'--')
        plt.plot([1e4/V_wave, 1e4/V_wave], [fm_curve.min(), fm_curve.max()] ,'--')
         
        # plot error snake
        x = info_dict['x']
        y = info_dict['y']
        CHI2 = info_dict['chi2'][i]
        CHI2_reduction = info_dict['chi2_reductions'][i]
        CHI2 /= CHI2_reduction
        CHI2 = CHI2 - np.min(CHI2)
         
        slo, shi = plot_snake(ax, ref_wave, fm_curve, redden_fm, x, y, CHI2)
         
        # plot power law reddening curve
        pl_red_curve = redden_pl2(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        plt.plot(ref_wave_inv, pl_red_curve, 'r-')
         
        # find 1-sigma and 2-sigma errors based on confidence
        maxebv_1sig, minebv_1sig = best_ebv, best_ebv
        maxrv_1sig, minrv_1sig = best_rv, best_rv
        for e, EBV in enumerate(x):
            for r, RV in enumerate(y):
                _chi2 = CHI2[r,e]
                if _chi2<1.00:
                    maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                    minebv_1sig = np.minimum(minebv_1sig, EBV)
                    maxrv_1sig = np.maximum(maxrv_1sig, RV)
                    minrv_1sig = np.minimum(minrv_1sig, RV)



### FORMAT SUBPLOT ###

# print data on subplot
        plttext = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
        plttext = plttext.format(best_ebv, maxebv_1sig-best_ebv, best_ebv-minebv_1sig,
                                 best_rv, maxrv_1sig-best_rv, best_rv-minrv_1sig
                                 )
            
        ax.text(.95, .98, plttext, size=INPLOT_LEGEND_FONTSIZE,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax.transAxes)
         
        # format subplot
        if i%PLOTS_PER_ROW == 0:
            ax.set_title('Phase: {}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel('$E(V-X)$', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_title('{}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
                     
            plt.xlim(1.0, 3.0)
            plt.ylim(-3.0, 2.0)
                
            labels = ax.get_yticks().tolist()
            labels[0] = labels[-1] = ''
            ax.set_yticklabels(labels)

            labels = ax.get_xticks().tolist()
            labels[0] = labels[-1] = ''
            ax.set_xticklabels(labels)
                                                     
            plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
            plt.setp(ax.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)


# format figure
        fig.suptitle('{}: Color Excess'.format(title), fontsize=TITLE_FONTSIZE)
    
        fig.text(0.5, .05, 'Inverse Wavelength ($1 / \mu m$)',
             fontsize=AXIS_LABEL_FONTSIZE, horizontalalignment='center')
         
        p1, = plt.plot(np.array([]), np.array([]), 'k--')
        p2, = plt.plot(np.array([]), np.array([]), 'r-')
        fig.legend([p1, p2], ['Fitzpatrick-Massa 1999*', 'Power-Law (Goobar 2008)'],
                    loc=1, bbox_to_anchor=(0, 0, .97, .99), ncol=2, prop={'size':LEGEND_FONTSIZE})
         
        fig.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.90, wspace=0.2, hspace=0.2)
        filenm = filter(str.isalnum, title)+'.png' # to get rid of white space and parentheses; and add extension.
        fig.savefig(filenm)

#plt.show()



if __name__=="__main__":
    
    # load spectra, interpolate 11fe to 12cu phases (only first 12)
    obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in obs_12cu]
        
        
    pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    art_reddened_11fe = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.0, rv=2.7, loadmast=False, loadptf=False))
    
        
    ########################
    # Choose 'SNobs' to be either an artificially reddened 11fe interpolated
    # to the phases of 12cu, or just choose 12cu itself.
    #
    obs_SN = art_reddened_11fe
    #
    ########################


    best_RVs, best_EBVs = grid_fit(phases, pristine_11fe, obs_SN)
    info_dict1 = cPickle.load(open("spectra_mag_fit_results_FILTERED.pkl", 'rb'))
    info_dict2 = cPickle.load(open("spectra_mag_fit_results_UNFILTERED.pkl", 'rb'))
                
    i = 0
    for t in zip(["SN2012cu (Feature Filtered)", "SN2012cu"], [info_dict1, info_dict2], pristine_11fe, obs_SN):
        if i > 0: break   # this is to not plot the unblocked fit.
        plot_excess(t[0], t[1], pristine_11fe, obs_SN)
        i += 1

