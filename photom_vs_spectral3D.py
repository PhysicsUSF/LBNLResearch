'''
::Author::
Xiaosheng Huang
(based on Andrew's plot_excess_contour.py)


Date: 12/8/14
Purpose: calculate effective AB mag for narrow bands equally spaced in inverse wavelength and compare that with the single-wavelength AB_nu (see below).  Also allow F99 to shift vertically.


There is A LOT of confusion about how to calculate synthetic photometry.  Here I summarize what is the two most important points from Bessell & Murphy
2012 (BM12):

1. One can calculate AB magnitude both from photon-counting and energy intergration -- see BM12 eq (A30).  As long as one uses the correct constants (which probably amounts to choosing the right zero points, which AS seems to have done correctly since the comparison with 2014J photometry is very good.)

2. The AB magnitude has a straightforward physical interpretation (BM12 eq (A29)):

                                AB mag = -2.5log<f_nu> - 48.557                                                         *
                                
    I can therefore back out <f_nu> given AB mag.
    
3. If the frequency bins are very small (or equivalently the inverse wavelength bins) are very small,

                                    <f_nu> --> f_nu                                                                     **
                                    
    And I can convert f_nu to f_lamb with BM12 eq (A1)
    
    
                                    f_lamb = f_nu*c/lamb^2                                                              ***
                                    
    I can then do the photometry fit and spectral fit comparison
    
    Or equivalently I can first convert f_lamb to f_nu, and then use BM12 eq (A3) to calculate 
    
                                AB_nu = -2.5log(f_nu) - 48.557                                                          ****
                                
    In the limit of very small nu (or inverse lamb) bins, my eqn **** should agree with my eq *.
    

(Bessell & Murphy 2012.  Use 48.577 to perfectly match Vega, which has a V mag of 0.03.  But if AB mag is
consistently used by me for single-wavelength mag, and by sncosmo for synthetic photometry (check) then
using 48.6 is just fine.)


- sncosmo's way of calculating the effective wavelength is in the following two lines in the class definition Bandpass() in Spectral.py:


    weights = self.trans * np.gradient(self.wave)
    return np.sum(self.wave * weights) / np.sum(weights)
    
    The gradient part will give a factor of 1 if the wavelengths are distributed 1 A apart.  Otherwise the weights are simply proportional to the transmission.  This is the same as BM12 eq (A14).

- I have changed 'vega' to 'ab' in loader.py.  1) to make easier comparison between photometric fit and spectral fit.  2) I ran into an error with 'vega' -- it seems that it can't retrieve data from a ftp site.  It really ought to be changed here and on the command line.

- Even with AB mag, I have recovered low RV values.  The last phase still looks different from other phases.

- At 40 bins or above, one need to remove nan to make it work.  - Dec 2. 2014



::Description::
This program will fit RV based on color excess of 2012cu

::Last Modified::
08/01/2014

'''
import argparse
from copy import deepcopy

import loader as l
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from itertools import izip
from loader import redden_fm, redden_pl, redden_pl2
from pprint import pprint
from scipy.stats import chisquare
from scipy.interpolate import interp1d

from plot_excess_contours import *  ## I may want to comment out this line, just make sure load_12cu_excess is the only function I use (which is now copies here.)
from mag_spectrum_fitting import ABmag_nu, extract_wave_flux_var, flux2mag, log, filter_features



### CONST ###
c = 3e18  # speed of light in A/sec.

### VARS ###



#EBV_GUESS = 1.1
#EBV_PAD = .3
#RV_GUESS = 2.75
#RV_PAD = .75


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15

PLOTS_PER_ROW = 6

V_wave = 5413.5


def get_excess(phases, select_phases, filters, pristine_11fe, obs_SN, mask, N_BUCKETS = -1, norm_meth = 'V_band'):
    

        

        del_lamb = 1.
        band_steps = 1200
        V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)


        EXCESS = {}
        EXCESS_VAR = {}

        DEL_MAG = {}
        DEL_MAG_VAR = {}

        for phase_index, phase in zip(select_phases, [phases[select_phases]]):
            #        for phase_index in phases: # [0,]
            
            
                print '\n\n\n Phase_index', phase_index, '\n\n\n'
            
            
                ref = pristine_11fe[phase_index]
                ## Note: I have determined that ref_wave is equally spaced at 2A.
                
                ## Even though ref_wave is defined here and is fed into extract_wave_flux_var(), it's not used for N-band fitting.
                ref_wave = ref[1].wave  # it is inefficient to define ref_wave in the for loop.  Should take it outside.  12/7/14.

                obs = obs_SN[phase_index]


                ref_flux = ref[1].flux
                obs_flux = obs[1].flux
                print 'obs_flux var estimated', np.var(obs_flux - ref_flux)
                #exit(1)



                ## mask for spectral features not included in fit


                ref_mag_norm, return_wave, ref_return_flux, ref_mag_avg_flux, ref_V_mag, ref_mag_var, ref_mag_V_var, ref_calib_err, nanmask_ref, _ \
                            = extract_wave_flux_var(ref_wave, ref, N_BUCKETS = N_BUCKETS, mask = mask, norm_meth = norm_meth)




                ## 12cu or artificially reddened 11fe
                obs_mag_norm, _, obs_return_flux, obs_mag_avg_flux, obs_V_mag, obs_mag_var, obs_mag_V_var, obs_calib_err, nanmask_obs, obs_flux \
                            = extract_wave_flux_var(ref_wave, obs, N_BUCKETS = N_BUCKETS, mask = mask, norm_meth = norm_meth)

                print 'flux var estimated:', np.var(obs_return_flux - ref_return_flux)
                print 'art_var', art_var
       
                print 'mag var estimated:', np.var(obs_mag_norm - ref_mag_norm)
                print 'avg obs_mag_var: %.3e' % (obs_mag_var.mean())
                print 'ref_mag_var', ref_mag_var.mean()
                print 'chi2/dof (in mag space) =', np.sum((obs_mag_norm - ref_mag_norm)**2/obs_mag_var)/(len(obs_mag_norm) - 2)

                plt.plot(return_wave, ref_mag_norm, 'k.')
                plt.plot(return_wave, obs_mag_norm, 'r.')
                plt.plot([return_wave.min(), return_wave.max()], [0, 0] ,'--')
                plt.plot([V_wave, V_wave], [obs_mag_norm.min(), obs_mag_norm.max()] ,'--')

                plt.show()

                #exit(1)

## Keep the following block for a bit longer; it's effective in diagnostics.  12/10/14
#                plt.figure()
#                plt.plot(return_wave, ref_mag_norm, 'k.')
#                plt.plot(return_wave, obs_mag_norm, 'r.')
# 
#                plt.figure()
#                plt.plot(return_wave, ref_mag_var, 'k.')
#                plt.plot(return_wave, obs_mag_var, 'r.')
#                plt.show()


                ## estimated of distance modulus
## ---> THIS CAN BE USED TO SHOW THAT V MAG DIFFERENCE IS A POOR INDICATOR FOR DISTANCE.  THIS MEANS TO USE V MAG NORMALIZATION TO TAKE OUT DISTANCE DIFFERENCE (AS FOLEY 2014 HAS DONE) IS NOT
## AN EFFECTIVE WAY. BUT THE MAG OF THE AVERAGE FLUX IS A CONSISTENT INDICATOR OF DISTANCE AND THAT'S WHY WE USE THIS AS THE NORMALIZATION TO TAKE OUT DISTANCE DIFFERENCE.
                del_mag_avg = obs_mag_avg_flux - ref_mag_avg_flux
                del_V_mag = obs_V_mag - ref_V_mag
                print '\n\n\n difference in magnitudes of average flux:', del_mag_avg
                print ' difference in V magnitudes:', del_V_mag, '\n\n\n'

                ## Total Variance.
                if norm_meth == 'AVG':
                    var = ref_mag_var + obs_mag_var  # NOT DEALING WITH BLOCKING FEATURES NOW 12/8/14
                    DEL_MAG[phase_index] = obs_mag_norm - ref_mag_norm
                    DEL_MAG_VAR[phase_index] = var
                elif norm_meth == 'V_band':
                    var = ref_mag_var + obs_mag_var # + obs_mag_V_var + ref_mag_V_var

                    V_mag_diff = obs_V_mag - ref_V_mag
                    print 'V_mag_diff', V_mag_diff
                    #exit(1)
                    
                    phase_excess = []
                    phase_var = []
                    for j, f in enumerate(filters):
                        phase_excess.append( (obs_mag_norm - ref_mag_norm)[j])  # Note since the mag's are normalized to V band, there's no need to subtract
                                                                                # off (obs_V_mag - ref_V_mag).
                        phase_var.append(var[j])
                    
                    EXCESS[phase_index] = phase_excess
                    EXCESS_VAR[phase_index] = phase_var

                print 'type phase_excess', type(phase_excess)
                plt.plot(return_wave, np.array(phase_excess), 'r.')
                plt.plot([return_wave.min(), return_wave.max()], [0, 0] ,'--')
                plt.plot([V_wave, V_wave], [np.array(phase_excess).min(), np.array(phase_excess).max()] ,'--')

                plt.show()
                #exit(1)

        if norm_meth == 'AVG':
            return DEL_MAG, DEL_MAG_VAR, return_wave
        elif norm_meth == 'V_band':
            return EXCESS, EXCESS_VAR, return_wave


def compute_sigma_level(L):
    """Adopted from: 
        
        https://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/
    """

    L[L == 0] = 1E-16  # this line redefines all 0 elements to be a small number.  It's pretty cool that the np.where()function is not really necessary.  The statment L==0 will generate the
                       # the needed indices.  Although I don't see how this statement is useful.
    logL = np.log(L)   # not sure the utility of this line.  may delete it soon.
    
    shape = L.shape
    L = L.ravel()   # this flattens the array.
    
    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)
    
    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    
    return L_cumsum[i_unsort].reshape(shape)


def plot_confidence_contours(ax, xdata, ydata, P, scatter=False, **kwargs):
    """Plot contours
        
        Adopted from:
            
            https://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/
        
        He also have a function for plotting the best-fit, with error snake, against data, plot_MCMC_model(ax, xdata, ydata, trace) -- I should look into and see if I can adopt it.
    
        
    """
    sigma = compute_sigma_level(P)
    ax.contour(xdata, ydata, sigma.T, levels=[0.683, 0.955], **kwargs)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')

    return sigma # it's really the CDF


def plot_contour3D(subplot_index, phase, red_law, excess, excess_var, wave,
                 ebv, ebv_pad, ebv_steps, rv, rv_pad, rv_steps, u_guess, u_pad, u_steps, ax=None):
    
    
    
        '''This is where chi2 is calculated. 
            
            I should separate the part that calculates chi2 and the part that plots.  
            
            -XH  12/6/14.
            
            
        '''
        

                         
        x = np.linspace(ebv-ebv_pad, ebv+ebv_pad, ebv_steps)
        y = np.linspace(rv-rv_pad, rv+rv_pad, rv_steps)
        
        X, Y = np.meshgrid(x, y)  ## <-------------- This part needs to be changed so that the steps for RV and EBV don't have to be the same: see how it's done in mag_spectrum_fitting.py


#        u_guess = 0
#        u_pad = 0.2
        ## Determine whether 2D or 3D fit.
        if u_steps > 1:
            ## 3D case
            u = np.linspace(u_guess - u_pad, u_guess + u_pad, u_steps)
            param_num = 3
        elif u_steps == 1:
            ## 2D case: only use the central value of u
            u = np.array([u_guess,])
            param_num = 2

        if ebv_steps == 1:
            x = np.array([ebv,])


        CHI2 = np.zeros((len(u), len(x), len(y)))

        #plt.figure()
        log( "Scanning CHI2 grid..." )
        for i, dist in enumerate(u):
            for j, EBV in enumerate(x):
                for k, RV in enumerate(y):
                    
                    ftz_curve = red_law(wave, np.zeros(wave.shape), -EBV, RV, return_excess=True)

                    print 'i, j, k', i, j, k
                    print 'dist, EBV, RV', dist, EBV, RV
                    print "reddening excess:", ftz_curve
                    print "12cu color excess:", excess - dist
                    
                    
                    #plt.plot(wave, ftz_curve, 'g.')
                    
                        
                    nanvals = np.isnan(excess)
                    nanmask = ~nanvals
                    #excess_var = excess_var[nanmask]
                    
                    CHI2[i, j, k] = np.sum( (((ftz_curve-excess) + dist)**2/excess_var)[nanmask])
         
        print 'CHI2: ', CHI2
        

        if np.sum(nanvals):
            print '\n\n\nWARNING. WARNGING. WARNTING.'
            print 'WARNING: THERE ARE %d BANDS WITH NAN VALUES.' % (np.sum(nanvals))
            print 'WARNING. WARNGING. WARNTING.\n\n\n'

                    
        print 'len(excess)', len(excess)

        
        dof = len(excess) - 1 - param_num  # degrees of freedom (V-band is fixed, N_BUCKETS-1 floating data pts).  I suppose even if we use mag_avg_flux as normalization we still lose 1 dof.

        CHI2_dof = CHI2/dof

        print 'CHI2_dof', CHI2_dof
        CHI2_dof_min = np.min(CHI2_dof)
        log("dof: {}".format(dof))
        log( "min CHI2: {}".format(np.min(CHI2)) )
        log( "min CHI2 per dof: {}".format(CHI2_dof_min) )

        delCHI2_dof = CHI2_dof - CHI2_dof_min

        mindex = np.where(delCHI2_dof == 0)   # Note argmin() only works well for 1D array.  -XH
        print 'mindex', mindex


        ####**** best fit values

        ## basically it's the two elements in mindex.  But each element is a one-element array; hence one needs an addition index of 0.
        mu, mx, my = mindex[0][0], mindex[1][0], mindex[2][0]
        print 'mindex', mindex
        print 'mu, mx, my', mu, mx, my
        best_u, best_rv, best_ebv = u[mu], y[my], x[mx]
        print 'best_u = %.3f, best_rv = %.3f, best_ebv = %.3f ' % (best_u, best_rv, best_ebv)
        ## estimate of distance modulus
        best_av = best_rv*best_ebv

        #plt.plot(wave, excess, 'r.')
        #plt.show()
        #exit(1)


        ####**** find 1-sigma and 2-sigma errors based on confidence


        best_fit_curve = redden_fm(wave, np.zeros(wave.shape), -best_ebv, best_rv, return_excess=True)
        snake_hi_1sig = deepcopy(best_fit_curve)
        snake_lo_1sig = deepcopy(best_fit_curve)

        maxebv_1sig, maxebv_2sig, minebv_1sig, minebv_2sig = best_ebv, best_ebv, best_ebv, best_ebv
        maxrv_1sig, maxrv_2sig, minrv_1sig, minrv_2sig = best_rv, best_rv, best_rv, best_rv
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
#                        snake_hi_1sig = np.maximum(snake_hi_1sig, redden_curve)  # the result is the higher values from either array is picked.
#                        snake_lo_1sig = np.minimum(snake_lo_1sig, redden_curve)

                    elif delCHI2_dof[i, e, r] < 4.00:
                        maxebv_2sig = np.maximum(maxebv_2sig, EBV)
                        minebv_2sig = np.minimum(minebv_2sig, EBV)
                        maxrv_2sig = np.maximum(maxrv_2sig, RV)
                        minrv_2sig = np.minimum(minrv_2sig, RV)




        print 'best_ebv, best_rv', best_ebv, best_rv
        print 'maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig', maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig
    
        
        ebv_uncert_upper = maxebv_1sig - best_ebv
        ebv_uncert_lower = best_ebv - minebv_1sig
        rv_uncert_upper = maxrv_1sig - best_rv
        rv_uncert_lower = best_rv - minrv_1sig
        
        print 'ebv_uncert_upper, ebv_uncert_lower', ebv_uncert_upper, ebv_uncert_lower
        print 'rv_uncert_upper, rv_uncert_lower', rv_uncert_upper, rv_uncert_upper
        

        log( "\t {}".format(mindex) )
        log( "\t u={} RV={} EBV={} AV={}".format(best_u, best_rv, best_ebv, best_av) )

## May need the following.
#        chi2dofs.append(CHI2_dof)
#        #chi2_reductions.append(CHI2_dof)
#        min_chi2s.append(CHI2_dof_min)
#        best_us.append(best_u)
#        best_rvs.append(best_rv)
#        best_ebvs.append(best_ebv)
#        best_avs.append(best_av)
#        
#        ebv_uncert_uppers.append(ebv_uncert_upper)
#        ebv_uncert_lowers.append(ebv_uncert_lower)
#        rv_uncert_uppers.append(rv_uncert_upper)
#        rv_uncert_lowers.append(rv_uncert_lower)

#        snake_lo_1sigs.append(snake_lo_1sig)
#        snake_hi_1sigs.append(snake_hi_1sig)




        # get best AV and calculate error in quadrature   # This is NOT the correct way. though the correct probably would give similar AV error since RV error dominates  -XH 12/2/14
        best_av = x[mx]*y[my]
        av_1sig = (best_av-np.sqrt((minebv_1sig-x[mx])**2 + (minrv_1sig-y[my])**2),
                   best_av+np.sqrt((maxebv_1sig-x[mx])**2 + (maxrv_1sig-y[my])**2))
                   
        av_2sig = (best_av-np.sqrt((minebv_2sig-x[mx])**2 + (minrv_2sig-y[my])**2),
                   best_av+np.sqrt((maxebv_2sig-x[mx])**2 + (maxrv_2sig-y[my])**2))
        


        ##  This is not the right way to handle CDF at all.  But I'm too lazy now.  Just trying to get CDF to be a 2D array.
        ##  When I add the variance to this calculation, I will get the contour plots right.



        P = np.sum(np.exp(-delCHI2_dof/2), axis = 0)  ## probability summed over the nuisance parameter u.  obviously un-normalized.
        ax = plt.subplot(111)
        CDF = plot_confidence_contours(ax, X, Y, P)
        print 'P > 1e-15:', P[P > 1e-15]
        print 'P > 0.1:', P[P > 0.1]
#plt.show()
#exit(1)

## Need to think about how to do contour plots -- basically what Zach and I went through in Starbucks in October.
#        if ax != None:
#                # plot contours
#                contour_levels = [0.0, 0.683, 0.955, 1.0]
#                plt.contourf(X, Y, 1-CDF, levels=[1-l for l in contour_levels], cmap=mpl.cm.summer, alpha=0.5)
#                C1 = plt.contour(X, Y, CDF, levels=[contour_levels[1]], linewidths=1, colors=['k'], alpha=0.7)
#                
#                #plt.contour(X, Y, CHISQ-chisq_min, levels=[1.0, 4.0], colors=['r', 'g'])
#                
#                # mark minimum
#                plt.scatter(x[mx], y[my], marker='s', facecolors='r')
#                
#                # show results on plot
#                if subplot_index%6==0:
#                        plttext1 = "Phase: {}".format(phase)
#                else:
#                        plttext1 = "{}".format(phase)
#                        
#                plttext2 = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
#                           "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
#                           "\n$A_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
#                plttext2 = plttext2.format(x[mx], maxebv_1sig-x[mx], x[mx]-minebv_1sig,
#                                           y[my], maxrv_1sig-y[my], y[my]-minrv_1sig,
#                                           best_av, av_1sig[1]-best_av, best_av-av_1sig[0]
#                                           )
#                                                   
#                if phase not in [11.5, 16.5, 18.5, 21.5]:
#                        ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
#                                horizontalalignment='left',
#                                verticalalignment='top',
#                                transform=ax.transAxes)
#                        ax.text(.04, .85, plttext2, size=INPLOT_LEGEND_FONTSIZE,
#                                horizontalalignment='left',
#                                verticalalignment='top',
#                                transform=ax.transAxes)
#                        ax.axhspan(2.9, (rv+rv_pad), facecolor='k', alpha=0.1)
#                
#                else:
#                        ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
#                                horizontalalignment='left',
#                                verticalalignment='top',
#                                transform=ax.transAxes)
#                        ax.text(.04, .32, plttext2, size=INPLOT_LEGEND_FONTSIZE,
#                                horizontalalignment='left',
#                                verticalalignment='top',
#                                transform=ax.transAxes)
#                        ax.axhspan(3.32, (rv+rv_pad), facecolor='k', alpha=0.1)
#                        ax.axhspan((rv-rv_pad), 2.5, facecolor='k', alpha=0.1)
#                        
#                        
#                # format subplot...
#                plt.ylim(rv-rv_pad, rv+rv_pad)
#                plt.xlim(ebv-ebv_pad, ebv+ebv_pad)
#                
#                ax.set_yticklabels([])
#                ax2 = ax.twinx()
#                ax2.set_xlim(ebv-ebv_pad, ebv+ebv_pad)
#                ax2.set_ylim(rv-rv_pad, rv+rv_pad)
#                
#                if subplot_index%6 == 5:
#                        ax2.set_ylabel('\n$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=5)
#                if subplot_index%6 == 0:
#                        ax.set_ylabel('$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=-2)
#                if subplot_index>=6:
#                        ax.set_xlabel('\n$E(B-V)$', fontsize=AXIS_LABEL_FONTSIZE)
#                
#                # format x labels
#                labels = ax.get_xticks().tolist()
#                labels[0] = labels[-1] = ''
#                ax.set_xticklabels(labels)
#                ax.get_xaxis().set_tick_params(direction='in', pad=-20)
#                
#                # format y labels
#                labels = ax2.get_yticks().tolist()
#                labels[0] = labels[-1] = ''
#                ax2.set_yticklabels(labels)
#                ax2.get_yaxis().set_tick_params(direction='in', pad=-30)
#                
#                plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
#                plt.setp(ax2.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)



## In terms of returned values, I don't think returning CDF is necessary for excess plot.  12/9/14

        return x, y, u, CDF, x[mx], y[my], u[mu], best_av, (minebv_1sig, maxebv_1sig), \
                                                 (minebv_2sig, maxebv_2sig), \
                                                 (minrv_1sig,  maxrv_1sig), \
                                                 (minrv_2sig,  maxrv_2sig), \
                                                 av_1sig, \
                                                 av_2sig


def plot_phase_excesses(name, EXCESS, filter_eff_waves, SN12CU_CHISQ_DATA, filters, red_law, phases, rv_spect, ebv_spect):
    
    
    print "Plotting excesses of",name," with best fit from contour..."
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    ## Keep; may need this later: pmin, pmax = np.min(phases), np.max(phases)
    
    #    for i, d, sn11fe_phase in izip(xrange(len(SN12CU_CHISQ_DATA)), SN12CU_CHISQ_DATA, sn11fe):
    for i, phase_index, phase, d in zip(range(len(SN12CU_CHISQ_DATA)), select_phases, [phases[select_phases]], SN12CU_CHISQ_DATA):


        print "Plotting phase {} ...".format(phase)
        
            
        ## KEEP, I will revert to this once this program has been thoroughly tested: ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        ax = plt.subplot(111)


        phase_excesses = np.array([EXCESS[phase_index][j] for j, f in enumerate(filters)])

        ## Keep the following two line in case I want to plot the symbols with different colors for different phases.  12/10/14
        #mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))
        #plt.plot(filter_eff_waves, phase_excesses, 's', color=mfc_color, ms=8, mec='none', mfc=mfc_color, alpha=0.8)

        plt.plot(filter_eff_waves, phase_excesses - best_u, 's', color='black', ms=8) #, mec='none', mfc=mfc_color, alpha=0.8)


        ## reddening law vars
        linestyle = '--'
        
        x = np.arange(3000,10000,10)
        #xinv = 10000./x
        red_curve = red_law(x, np.zeros(x.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)
        plt.plot(x, red_curve, 'k'+linestyle)
        slo, shi = plot_snake(ax, x, red_curve, red_law, d['x'], d['y'], d['CDF'])

        red_curve_spect = red_law(x, np.zeros(x.shape), -ebv_spect, rv_spect, return_excess=True)
        plt.plot(x, red_curve_spect, 'r-')

        ## plot where V band is.   -XH
        plt.plot([x.min(), x.max()], [0, 0] ,'--')
        plt.plot([V_wave, V_wave], [red_curve.min(), red_curve.max()] ,'--')




        #pprint( zip([int(f) for f in filter_eff_waves],
                #[round(f,2) for f in 10000./np.array(filter_eff_waves)],
                #filters,
                #[round(p,2) for p in phase_excesses],
                #[round(r,2) for r in shi(filter_eff_waves)],
                #[round(r,2) for r in slo(filter_eff_waves)],
                #[round(r,2) for r in interp1d(x, test_red_curve)(filter_eff_waves)]
                #)
               #)
        
        
        ## print data on subplot
        plttext = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$A_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$u={:.2f}$" + "\n$R_V(sp) = {:.2f}$" + "\n$E(B-V)(sp) = {:.2f}$"
                  


        print 'best_u', d['BEST_u']

        plttext = plttext.format(d['BEST_EBV'], d['EBV_1SIG'][1]-d['BEST_EBV'], d['BEST_EBV']-d['EBV_1SIG'][0],
                                 d['BEST_RV'], d['RV_1SIG'][1]-d['BEST_RV'], d['BEST_RV']-d['RV_1SIG'][0],
                                 d['BEST_AV'], d['AV_1SIG'][1]-d['BEST_AV'], d['BEST_AV']-d['AV_1SIG'][0],
                                 d['BEST_u'], rv_spect, ebv_spect)
            


        ax.text(.95, .3, plttext, size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        
        ## format subplot
        if i%PLOTS_PER_ROW == 0:
            ax.set_title('Phase: {}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel('$E(V-X)$', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_title('{}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
    
        plt.xlim(3000, 10000)
        plt.ylim(-3.0, 2.0)
        
        labels = ax.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_yticklabels(labels)
        
        labels = ax.get_xticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_xticklabels(labels)
        
        plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
        plt.setp(ax.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)


def plot_snake(ax, wave, init, red_law, x, y, CDF, plot2sig=False):
    snake_hi_1sig = deepcopy(init)
    snake_lo_1sig = deepcopy(init)
    if plot2sig:
        snake_hi_2sig = deepcopy(init)
        snake_lo_2sig = deepcopy(init)
    
    for i, EBV in enumerate(x):
        for j, RV in enumerate(y):
            if CDF[j,i]<0.683:
                red_curve = red_law(wave, np.zeros(wave.shape), -EBV, RV, return_excess=True)
                snake_hi_1sig = np.maximum(snake_hi_1sig, red_curve)
                snake_lo_1sig = np.minimum(snake_lo_1sig, red_curve)
            elif plot2sig and CDF[j,i]<0.955:
                red_curve = red_law(wave, np.zeros(wave.shape), -EBV, RV, return_excess=True)
                snake_hi_2sig = np.maximum(snake_hi_2sig, red_curve)
                snake_lo_2sig = np.minimum(snake_lo_2sig, red_curve)
    
    ax.fill_between(wave, snake_lo_1sig, snake_hi_1sig, facecolor='black', alpha=0.3)
    if plot2sig:
        ax.fill_between(wave, snake_lo_2sig, snake_hi_2sig, facecolor='black', alpha=0.1)
    
    return interp1d(wave, snake_lo_1sig), interp1d(wave, snake_hi_1sig)


def simulate_11fe(phases_12cu, no_var = True, art_var = 1e-60, ebv = -1.0, rv = 2.8, del_mu = 0.0):

    '''
    simulate reddened 11fe, with its original variances removed and articial variances (initially, uniform) added.
    Also for pristine_11fe its variances are removed (at least initially).
    '''

#    pristine_11fe = l.get_11fe(loadmast=False, loadptf=False)
#    art_reddened_11fe = l.get_11fe('fm', ebv=ebv, rv=rv, art_var=art_var, loadmast=False, loadptf=False)



    #print 'phases in simulate_11fe', phases
    #exit(1)
    #pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(no_var = no_var, loadmast=False, loadptf=False))
    #art_reddened_11fe = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=ebv, rv=rv, art_var=art_var, loadmast=False, loadptf=False))
    #exit(1)

#    print 'var_th', art_reddened_11fe[0][1].error.mean()
#    
#    ref_flux = pristine_11fe[0][1].flux
#    obs_flux = art_reddened_11fe[0][1].flux
#    print 'var est (3)', np.var(obs_flux - ref_flux)
#    exit(1)

    phases_12cu = np.array(phases_12cu)
    pristine_11fe = l.nearest_spectra(phases_12cu, l.get_11fe(no_var = no_var, loadmast=False, loadptf=False))
    #pristine_11fe = l.get_11fe_nearest_phase(phases_12cu, redtype=None, ebv=None, rv=None, av=None, p=None, del_mu=0.0, no_var = no_var, art_var = 0, loadsnf=True)
    #pristine_11fe = l.get_11fe(no_var = no_var, loadmast=False, loadptf=False)
    print 'artificially redden 11fe with ebv, rv = ', ebv, rv
    #exit(1)
    art_reddened_11fe = l.nearest_spectra(phases_12cu, l.get_11fe('fm', ebv=ebv, rv=rv, art_var=art_var, loadmast=False, loadptf=False))

    phases_11fe = [t[0] for t in pristine_11fe]
    phases_reddened_11fe = [t[0] for t in art_reddened_11fe]

    print 'Fetched 11fe phases:', phases_11fe
    print 'Reddened 11fe phases:', phases_reddened_11fe

    return pristine_11fe, art_reddened_11fe


if __name__ == "__main__":

    '''
        
    
    python photom_vs_spectral3D.py -obs_SN '12cu' -select_phases 0 -N_BUCKETS 20 -u_guess 0.0 -u_pad 0.2 -u_steps 21 -EBV_GUESS 1.0 -EBV_PAD 0.3 -EBV_STEPS 41 -RV_GUESS 2.8 -RV_PAD 1.0 -RV_STEPS 41 -ebv_spect 1.00 -rv_spect 2.8 -art_var 1e-31 -unfilt
    
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-N_BUCKETS', type = int)
    parser.add_argument('-obs_SN', type = str)
    parser.add_argument('-RV_GUESS', type = float)
    parser.add_argument('-RV_PAD', type = float)
    parser.add_argument('-RV_STEPS', type = int)
    parser.add_argument('-EBV_GUESS', type = float)
    parser.add_argument('-EBV_PAD', type = float)
    parser.add_argument('-EBV_STEPS', type = int)
    parser.add_argument('-u_guess', type = float)
    parser.add_argument('-u_pad', type = float)
    parser.add_argument('-u_steps', type = int)
    parser.add_argument('-art_var', type = float)
    
    
    parser.add_argument('-select_phases',  '--select_phases', nargs='+', type=int)  # this can take a tuple: -select_phases 0 4  but the rest of the program can't handle more than
                                                                                    # one phases yet.  -XH 12/7/14
    _ = parser.add_argument('-ebv_spect', type = float)  # just another way to add an argument to the list.
    _ = parser.add_argument('-rv_spect', type = float)  # just another way to add an argument to the list.
    _ = parser.add_argument('-unfilt', '--unfilt', action='store_true')  # just another way to add an argument to the list.


    args = parser.parse_args()
    print 'args', args
    obs_SN = args.obs_SN
    N_BUCKETS = args.N_BUCKETS
    RV_GUESS = args.RV_GUESS
    RV_PAD = args.RV_PAD
    RV_STEPS = args.RV_STEPS
    EBV_GUESS = args.EBV_GUESS
    EBV_PAD = args.EBV_PAD
    EBV_STEPS = args.EBV_STEPS
    u_guess = args.u_guess
    u_pad = args.u_pad
    u_steps = args.u_steps
    select_phases = np.array(args.select_phases) ## if there is only one phase select, it needs to be in the form of a 1-element array for all things to work.
    ebv_spect = args.ebv_spect
    rv_spect = args.rv_spect
    art_var = args.art_var
    unfilt = args.unfilt

    hi_wave = 9700.
    lo_wave = 3300.
    


    
    ## load spectra, interpolate 11fe to 12cu phases (only first 11)
    obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:11]
    phases_12cu = [t[0] for t in obs_12cu]
    print 'Available 12cu phases', phases_12cu
    #exit(1)
 
 
    ## obs_SN is either an artificially reddened 11fe interpolated to the phases of 12cu, or 12cu itself.
    if obs_SN == 'red_11fe':
        pristine_11fe, obs_SN = simulate_11fe(phases_12cu, no_var = True, art_var = art_var, ebv = -EBV_GUESS, rv = RV_GUESS, del_mu = 0.0)
    elif obs_SN == '12cu':
        obs_SN = obs_12cu
        pristine_11fe = l.interpolate_spectra(phases_12cu, l.get_11fe(loadmast=False, loadptf=False))



    ref_wave = pristine_11fe[0][1].wave   ## this is not the most elegant way of doing things.  I have an identical statement in get_excess().  need to make this tighter.


    if unfilt == True:
        FEATURES_ACTUAL = []
    else:
        FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),\
                          (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]


    mask = filter_features(FEATURES_ACTUAL, ref_wave)

    ref_wave = ref_wave[mask]   ## this is not the most elegant way of doing things.  I have an identical statement in get_excess().  need to make this tighter.



    ## Setting up tophat filters
    filters_bucket, zp_bucket, LOW_wave, HIGH_wave = l.generate_buckets(lo_wave, hi_wave, N_BUCKETS)  #, inverse_microns=True)
    filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff for f in filters_bucket])

    del_wave = (HIGH_wave  - LOW_wave)/N_BUCKETS



    if N_BUCKETS > 0:
        filters = filters_bucket
    else:
        filters = []
        for i, wave in enumerate(ref_wave):
            if i == np.argmin(abs(wave - V_wave)):
                filters.append('V')
            else:
                filters.append(i)


    EXCESS, EXCESS_VAR, wave = get_excess(phases_12cu, select_phases, filters, pristine_11fe, obs_SN, mask, N_BUCKETS = N_BUCKETS, norm_meth = 'V_band')



    fig = plt.figure(figsize = (10, 8))
    SN12CU_CHISQ_DATA = []
    for phase_index, phase in zip(select_phases, [phases_12cu[select_phases]]):

        print "Plotting phase {} ...".format(phase)
        
        contour_ax = plt.subplot(111)  # ax = plt.subplot(2,6,i+1)
        
        
        ## plot_contour3D() is where chi2 is calculated.
 
        x, y, u, CDF, \
        best_ebv, best_rv, best_u, best_av, \
            ebv_1sig, ebv_2sig, \
            rv_1sig, rv_2sig, \
            av_1sig, av_2sig = plot_contour3D(phase_index, phase, redden_fm, EXCESS[phase_index], EXCESS_VAR[phase_index],
                                            wave, EBV_GUESS,
                                            EBV_PAD, EBV_STEPS, RV_GUESS, RV_PAD, RV_STEPS, u_guess, u_pad, u_steps, contour_ax
                                            )
        
        SN12CU_CHISQ_DATA.append({'phase'   : phase,
                                 'x'       : x,
                                 'y'       : y,
                                 'u'       : u,
                                 'CDF'     : CDF,
                                 'BEST_EBV': best_ebv,
                                 'BEST_RV' : best_rv,
                                 'BEST_u'  : best_u,
                                 'BEST_AV' : best_av,
                                 'EBV_1SIG': ebv_1sig,
                                 'EBV_2SIG': ebv_2sig,
                                 'RV_1SIG' : rv_1sig,
                                 'RV_2SIG' : rv_2sig,
                                 'AV_1SIG' : av_1sig,
                                 'AV_2SIG' : av_2sig
                                 })


    fig.subplots_adjust(left=0.04, bottom=0.08, right=0.95, top=0.92, hspace=.06, wspace=.1)
    fig.suptitle('SN2012CU: $E(B-V)$ vs. $R_V$ Contour Plot per Phase', fontsize=TITLE_FONTSIZE)

    fig = plt.figure(figsize = (20, 12))
    plot_phase_excesses('SN2012CU', EXCESS, wave, SN12CU_CHISQ_DATA, filters, redden_fm, phases_12cu, rv_spect, ebv_spect)

    plt.show()

