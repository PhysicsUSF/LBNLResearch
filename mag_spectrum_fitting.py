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

V_wave = 5413.5  # see my mag_spectrum_plot_excess.py
FrFlx2mag = 2.5/np.log(10)  #  =1.0857362


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

    flux = flux_interp(ref_wave)

    var = interp1d(SN[1].wave, var)(ref_wave)


    # B-V color for 11fe
    #ref_B_V = -2.5*np.log10(ref_flux[np.abs(ref_wave - 4400).argmin()]/ref_flux[np.abs(ref_wave - 5413.5).argmin()])
    #print 'B-V for 11fe:', ref_B_V
                
    calib_err_mag = SN[2]


#    ref_interp = interp1d(ref_wave, ref_flux)   # What does this accomplish?  -XH 11/25/14
    
    ## convert flux, variance, and calibration error to magnitude space
    
    mag_norm, mag_var = flux2mag(flux, flux_interp, var, norm_meth = norm_meth)
    
    
    if mask != None:
        mag_norm = mag_norm[mask]  # Note: mask has the same length as mag_norm, and contains a bunch of 0's and 1's (the 0's are where the blocked features are).
                                   # This is a very pythonic way of doing things: even though mask doesn't specifiy the indices of the wavelengths that should
                                   # be blocked, the operation mag_norm[mask] does just that.  One can think of mask as providing a truth table that tells python
                                   # which of the elements in mag_norm to keep and which to discard.  Yes, it doesn't make sense at first sight since mask doesn't
                                   # contain indices.  But it does work, and is the pythonic way!  -XH 11/25/14.


    # get mask for nan-values
    nanmask = ~np.isnan(mag_norm)
    

    return mag_norm, mag_var, calib_err_mag, nanmask, flux


def flux2mag(flux, flux_interp=None, var=None, norm_meth = 'AVG'):
    mag_var = None
    
    mag = -2.5*np.log10(flux)
    
    if norm_meth == 'AVG':
        mag_zp = -2.5*np.log10(np.average(flux))  # One shouldn't use the photon noise as the weight to find the average flux - see NB 11/22/14.
                                                  # Note it's not the avg mag but the mag of the avg flux.
    elif norm_meth == 'single_V':
        mag_zp = -2.5*np.log10(flux_interp(V_wave))

    mag_norm = -(mag - mag_zp)  # the minus sign is because we will plot E(V-X)

    # calculate magnitude error
    if type(var)!=type(None):
        fr_err = np.sqrt(var)/flux
        mag_var = (FrFlx2mag*fr_err)**2


    results = tuple([r for r in (mag_norm, mag_var) if type(r)!=type(None)])

    return (results, results[0])[len(results)==1]  # the purpose of this statement is that if only results[0] needs to be returned then that will happen.


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
        chi2s = []
        min_chi2s = []
        chi2_reductions = []
        
        #V_band = [(5300., 5500., 'Vband')]

        del_lamb = 1.
        band_steps = 1200
        V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)


        for phase_index in xrange(len(phases)): # [0,]: # xrange((len(phases)):
            
            
                print '\n\n\n Phase_index', phase_index, '\n\n\n'
            
                ref = pristine_11fe[phase_index]
                ref_wave = ref[1].wave

                obs = obs_SN[phase_index]


                # mask for spectral features not included in fit
                mask = filter_features(FEATURES_ACTUAL, ref_wave)


                ref_mag_norm, ref_mag_var, ref_calib_err, nanmask_ref, _ = extract_wave_flux_var(ref_wave, ref, mask = mask, norm_meth = 'AVG')

#                        print 'ref_mag_norm', len(ref_mag_norm)
#                        print 'ref_wave', len(ref_wave)
#                        print 'mask', len(mask)
#                        
#                        exit(1)
#                        

                log()
                log( "Phase: {}".format(ref[0]) )
                


                ## 12cu/reddened 11fe

                obs_mag_norm, obs_mag_var, obs_calib_err, nanmask_obs, obs_flux = extract_wave_flux_var(ref_wave, obs, mask = mask, norm_meth = 'AVG')


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

                #################################################
                
                
                
                ## for calculation of CHI2 per dof
                CHI2_reduction = np.sum(nanmask) - 2  # (num. data points)-(num. parameters)
                log( "CHI2 reduction: {}".format(CHI2_reduction) )
                
                #################################################
                
                if u_steps > 1:
                    u = np.linspace(u_guess - u_pad, u_guess + u_pad, u_steps)
                elif u_steps == 1:
                    u = np.array([u_guess,])

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
                                unred_mag_norm = flux2mag(unred_flux, norm_meth = 'AVG')
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




                ### report/save results
                
                #print 'CHI2', CHI2
                
                chi2_min = np.min(CHI2)
                log( "min chi2: {}".format(chi2_min) )
                
                mindex = np.where(CHI2==chi2_min)   # can try use argmin() here.  -XH
                
                # basically it's the two elements in mindex.  But each element is a one-element array; hence one needs an addition index of 0.
                mu, mx, my = mindex[0][0], mindex[1][0], mindex[2][0]
                print 'mindex', mindex
                print 'mu, mx, my', mu, mx, my
                print 'best_u, best_rv, best_ebv', u[mu], x[mx], y[my]
               
               
               #exit(1)
                
                log( "\t {}".format(mindex) )
                log( "\t u={} RV={} EBV={} AV={}".format(u[mu], y[my], x[mx], x[mx]*y[my]) )
                
                chi2s.append(CHI2)
                chi2_reductions.append(CHI2_reduction)
                min_chi2s.append(chi2_min)
                best_us.append(u[mu])
                best_rvs.append(y[my])
                best_ebvs.append(x[mx])
                best_avs.append(x[mx]*y[my])

#        return best_rvs, best_ebvs

        pprint( zip(phases, best_rvs, best_ebvs, best_avs, min_chi2s) )
                
        ## save results with date
        #                filename = "spectra_mag_fit_results_{}.pkl".format(strftime("%H-%M-%S-%m-%d-%Y", gmtime()))

        filename = "spectra_mag_fit_results_FILTERED.pkl"
#        cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
#                        'chi2': chi2s, 'chi2_reductions': chi2_reductions, 'u_steps': u_steps, 'rv_steps': rv_steps, 'ebv_steps': ebv_steps,
#                        'u': u, 'x': x, 'y': y, 'X': X, 'Y': Y},
#                        open(filename, 'wb'))

        cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                     'chi2': chi2s, 'chi2_reductions': chi2_reductions, 'u_steps': u_steps, 'rv_steps': rv_steps, 'ebv_steps': ebv_steps,
                     'u': u, 'x': x, 'y': y}, open(filename, 'wb'))
        
        log( "Results successfully saved in: {}".format(filename) )



#best_rvs, best_ebvs = perphase_fit()
        print 'in per_phase():', best_us, best_rvs, best_ebvs
        #        print 'in per_phase():', type(best_rvs), type(best_ebvs)

#exit(1)
        return best_us, best_rvs, best_ebvs



def plot_excess(title, info_dict, pristine_11fe, obs_SN):
    
    fig = plt.figure(figsize = (20, 12))
    
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
    V_wave = np.interp(0., f99ebv, f99wv)  # need to add comment here -- see comment immediately below.  -XH 11/25/14
    #V_wave = 5413.5  # See the comment for the above block of code.
    # AS previously used V_wave = 5417.2
    
    
    
    
    for i, phase in enumerate(phases):
        
        
        print "Plotting phase {} ...".format(phase)
        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        
        ref = pristine_11fe[i]
        obs = obs_SN[i]
        
        best_ebv = info_dict['ebv'][i]
        best_rv  = info_dict['rv'][i]
        
        ref_wave = ref[1].wave

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
        ref_wave_inv = 10000./ref_wave
        mfc_color = plt.cm.cool(5./11)
        
        # plot excess
        plt.plot(ref_wave_inv, excess, '.', color=mfc_color, ms=6, mec='none', mfc=mfc_color, alpha=0.8)



        # plot reddening curve
        fm_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        fm_curve27 = redden_fm(ref_wave, np.zeros(ref_wave.shape), -1.02*np.ones(best_ebv.shape), 2.7*np.ones(best_rv.shape), return_excess=True)
        plt.plot(ref_wave_inv, fm_curve, 'k--')
        plt.plot(ref_wave_inv, fm_curve27, 'r-')
        
        # plot where V band is.   -XH
        plt.plot([ref_wave_inv.min(), ref_wave_inv.max()], [0, 0] ,'--')
        plt.plot([1e4/V_wave, 1e4/V_wave], [fm_curve.min(), fm_curve.max()] ,'--')
         
        ## plot error snake

        u = info_dict['u']  # this is the distance dimension of the fitting.

        x = info_dict['x']
        y = info_dict['y']
        CHI2 = info_dict['chi2'][i]
        CHI2_reduction = info_dict['chi2_reductions'][i]
        CHI2 /= CHI2_reduction
        delCHI2 = CHI2 - np.min(CHI2)
         
         #slo, shi = plot_snake(ax, ref_wave, fm_curve, redden_fm, x, y, CHI2)
         
        # plot power law reddening curve
        #pl_red_curve = redden_pl2(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        #plt.plot(ref_wave_inv, pl_red_curve, 'r-')
         
        # find 1-sigma and 2-sigma errors based on confidence
        maxebv_1sig, minebv_1sig = best_ebv, best_ebv
        maxrv_1sig, minrv_1sig = best_rv, best_rv
        for i, u in enumerate(u):
            for e, EBV in enumerate(x):
                for r, RV in enumerate(y):
                    del_chi2 = delCHI2[i,r,e]
                    if del_chi2<1.00:
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


# format figure
        fig.suptitle('{}: Color Excess'.format(title), fontsize=TITLE_FONTSIZE)
    
        fig.text(0.5,
                 .05, 'Inverse Wavelength ($1 / \mu m$)',
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
    
    # load spectra, interpolate 11fe to 12cu phases (only first 12)
    obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in obs_12cu]
        
        
    pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    art_reddened_11fe = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.0, rv=2.8, loadmast=False, loadptf=False))
    
        
    ########################
    # Choose 'SNobs' to be either an artificially reddened 11fe interpolated
    # to the phases of 12cu, or just choose 12cu itself.
    #
    #obs_SN = art_reddened_11fe
    obs_SN = obs_12cu
    #
    ########################


    best_us, best_rvs, best_ebvs = grid_fit(phases, pristine_11fe, obs_SN, u_guess=0., u_pad=0.15, u_steps = 61, rv_guess=2.8, rv_pad=0.5, rv_steps=101, ebv_guess=1.0, ebv_pad=0.2, ebv_steps = 101)
    info_dict1 = cPickle.load(open("spectra_mag_fit_results_FILTERED.pkl", 'rb'))
    info_dict2 = cPickle.load(open("spectra_mag_fit_results_UNFILTERED.pkl", 'rb'))
                
    i = 0
    for t in zip(["SN2012cu (Feature Filtered)", "SN2012cu"], [info_dict1, info_dict2], pristine_11fe, obs_SN):
        if i > 0: break   # this is to not plot the unblocked fit.
        plot_excess(t[0], t[1], pristine_11fe, obs_SN)
        i += 1

