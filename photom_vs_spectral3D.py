'''
::Author::
Xiaosheng Huang
(based on Andrew's plot_excess_contour.py)


Original Date: 12/8/14
Last Updated: 12/17/14
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

- Even with AB mag, I have recovered low RV values.  The last phase still looks different from other phases -- this is now understood: the last phase of 12cu at day 26.5 was being matched by the 11fe phase of 23.7 and 74.x, which of course makes doing linear interpolation ridiculous.  We had realized this problem in the summer and it was solved after we incorporated other data sets (downloaded from mast and ptf), one of which has a phase much closer to 26.5.  -Dec 17, 2014

- At 40 bins or above, one need to remove nan to make it work.  - Dec 2. 2014


- I have now realized the error that linearly interpolating variance is not the right thing to do.  When one adds two quantities to get a third quantity (what interpolation does), there are well-understood rules governing the variance of the third quantity based on the variances of the first two (Bevington eq (3.20)).  However when one interpolates and calculates the variance according to Bevington, if the new set of points interlace with the original set of points, then obviously the uncertainties of the new set of points are correlated.  This is why it's a tricky problem.  Standard packages (pysynphot, e.g.) can interpolate fluxes but don't deal with variances!  One way to get around that may be Gaussian Process: http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_regression.html.

   But I won't try a complicated tool like that for now.  Instead I will bin 12cu and 11fe spectral data into 1000 common bins.  There should be no ambiguity as to how the variance of each bin should be calculated.
   
   And I avoid interpolate between phases by choosing the phases of 11fe that match 12cu phases the closest.   Dec 17, 2014


::Description::
This program will fit RV based on color excess of 2012cu

::Last Modified::
12/17/2014

'''
import argparse
from copy import deepcopy
import math
import pickle


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
from scipy.optimize import minimize
from scipy.optimize import curve_fit



from mag_spectrum_fitting import ABmag_nu, extract_wave_flux_var, flux2mag, log, filter_features
import plots as plots


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



V_wave = 5413.5  # the wavelength at which F99's excess curve is zero.


def get_excess(phases, select_phases, filters, pristine_11fe, obs_SN, mask, N_BUCKETS = -1, norm_meth = 'V_band'):
        
    '''

        
    '''
    
    del_lamb = 1.
    band_steps = 1200
    ## this statement is currently unused -- consider deleting it.  1/16/15
    V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)


    EXCESS = {}
    EXCESS_VAR = {}

    V_MAG_DIFF = {}
    DEL_MAG = {}
    DEL_MAG_VAR = {}
    

    for phase_index, phase in zip(select_phases, [phases[i] for i in select_phases]):        
        
        print '\n\n\nPhase_index', phase_index
        print 'Phase', phase, '\n\n\n'
    
    
        ref = pristine_11fe[phase_index]
        
        obs = obs_SN[phase_index]


        ## mask for spectral features not included in fit


        ref_mag_norm, return_wave, ref_return_flux, ref_return_flux_var, ref_mag_avg_flux, ref_V_mag, ref_mag_var, _ , _ \
                    = extract_wave_flux_var(ref, N_BUCKETS = N_BUCKETS, norm_meth = norm_meth)


        ## 12cu or artificially reddened 11fe
        obs_mag_norm, _, obs_return_flux, obs_return_flux_var, obs_mag_avg_flux, obs_V_mag, obs_mag_var, _ , _ \
                    = extract_wave_flux_var(obs, N_BUCKETS = N_BUCKETS, norm_meth = norm_meth)

        return_wave = return_wave[mask]
        
        ref_return_flux = ref_return_flux[mask]
        ref_mag_norm = ref_mag_norm[mask]
        ref_mag_var = ref_mag_var[mask]
        ref_return_flux_var = ref_return_flux_var[mask]

        obs_return_flux = obs_return_flux[mask]
        obs_mag_norm = obs_mag_norm[mask]
        obs_mag_var = obs_mag_var[mask]
        obs_return_flux_var = obs_return_flux_var[mask]


            
## Diagnostic printing statements
#        print 'flux var estimated:', np.var(obs_return_flux - ref_return_flux)
#        print 'art_var', art_var
#
#        print 'mag var estimated:', np.var(obs_mag_norm - ref_mag_norm)
#        print 'avg obs_mag_var: %.3e' % (obs_mag_var.mean())
#        print 'ref_mag_var', ref_mag_var.mean()
#        print 'chi2/dof (in mag space) =', np.sum((obs_mag_norm - ref_mag_norm)**2/(ref_mag_var + obs_mag_var))/(len(obs_mag_norm) - 2)

## All plots below are diagnostic.  Need to be deleted soon.
#        plt.figure()
#        plt.plot(return_wave, ref_return_flux, 'k.')
#        plt.errorbar(return_wave, obs_return_flux, np.sqrt(obs_return_flux_var), fmt='r.')
#        plt.title('Spectrum in flux space')
#
#        plt.figure()
#        plt.plot(return_wave, ref_mag_norm, 'k.')
#        plt.errorbar(return_wave, obs_mag_norm, np.sqrt(obs_mag_var), fmt='r.')
#        plt.plot([return_wave.min(), return_wave.max()], [0, 0] ,'--')
#        plt.plot([V_wave, V_wave], [obs_mag_norm.min(), obs_mag_norm.max()] ,'--')
#        plt.title('Spectrum in mag space (normalized to V)')
        

## Keep the following block for a bit longer; it's effective in diagnostics.  12/10/14
#                plt.figure()
#                plt.plot(return_wave, ref_mag_norm, 'k.')
#                plt.plot(return_wave, obs_mag_norm, 'r.')
# 
#                plt.figure()
#                plt.plot(return_wave, ref_mag_var, 'k.')
#                plt.plot(return_wave, obs_mag_var, 'r.')
#                plt.show()


        ## estimated of fluctance modulus
## ---> THIS CAN BE USED TO SHOW THAT V MAG DIFFERENCE IS A POOR INDICATOR FOR DISTANCE.  THIS MEANS TO USE V MAG NORMALIZATION TO TAKE OUT DISTANCE DIFFERENCE (AS FOLEY 2014 HAS DONE) IS NOT
## AN EFFECTIVE WAY. BUT THE MAG OF THE AVERAGE FLUX IS A CONSISTENT INDICATOR OF DISTANCE AND THAT'S WHY WE USE THIS AS THE NORMALIZATION TO TAKE OUT DISTANCE DIFFERENCE.
        del_mag_avg = obs_mag_avg_flux - ref_mag_avg_flux
        del_V_mag = obs_V_mag - ref_V_mag
        print '\n\n\n difference in magnitudes of average flux:', del_mag_avg
        print ' difference in V magnitudes:', del_V_mag, '\n\n\n'


        ## I think whether norm_meth is "AVG" or "V_band" the treatment below should be the same and thus the if..elif statement below is NOT necessary. 
        ## Total Variance.
#        if norm_meth == 'AVG':
#            var = ref_mag_var + obs_mag_var  # NOT DEALING WITH BLOCKING FEATURES NOW 12/8/14
#            DEL_MAG[phase_index] = obs_mag_norm - ref_mag_norm
#            DEL_MAG_VAR[phase_index] = var
#        elif norm_meth == 'V_band':
        var = ref_mag_var + obs_mag_var # + obs_mag_V_var + ref_mag_V_var -- apparently to get chi2/dof being 1, I don't need to add
                                            # obs_mag_V_var and ref_mag_V_var.  In fact if I do, I get chi2/dof ~0.5.  Need to think this thru.

            
        phase_excess = []
        phase_var = []

#### --------> It's unclear to me why the for loop below is necessary -- why can't I just do phase_excess = obs_mag_norm - ref_mag_norm and phase_var = var?  1/16/15  <---------------

        for j, f in enumerate(filters):
#                print 'j, f', j, f
#                print len(filters)
#                print len(obs_mag_norm)
#                print len(ref_mag_norm)
#                exit(1)
            phase_excess.append( (obs_mag_norm - ref_mag_norm)[j])  # Note since the mag's are normalized to V band, there's no need to subtract
                                                                        # off (obs_V_mag - ref_V_mag).
            phase_var.append(var[j])
            
        EXCESS[phase_index] = np.array(phase_excess)
        EXCESS_VAR[phase_index] = np.array(phase_var)
        V_MAG_DIFF[phase_index] = del_V_mag

        #plt.plot(return_wave, np.array(phase_excess), 'r.')
#        plt.figure()
#        plt.errorbar(return_wave, np.array(phase_excess), np.array(phase_var), fmt='r.', label=u'excess (w/o u)') #, 's', color='black', ms=8) #, mec='none', mfc=mfc_color, alpha=0.8)

#    if norm_meth == 'AVG':
#        return DEL_MAG, DEL_MAG_VAR, return_wave
#    elif norm_meth == 'V_band':
    return EXCESS, EXCESS_VAR, return_wave, V_MAG_DIFF


def compute_sigma_level(L):
    """
        Input: probability 
        Returns: CDF
        
        Adopted from: 
        
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


## This function doesn't seem to be called anywhere.  Can delete soon.  1/7/15
#
#def plot_confidence_contours(ax, xdata, ydata, CDF, scatter=False, **kwargs):
#    """Plot contours
#        
#        Adopted from:
#            
#            https://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/
#        
#        He also have a function for plotting the best-fit, with error snake, against data, plot_MCMC_model(ax, xdata, ydata, trace) -- I should look into and see if I can adopt it.
#    
#        
#    """
#    #CDF = compute_sigma_level(P).T
#
#
#    contour_levels = [0.0, 0.683, 0.955, 1.0]
#    ## plots 1- and 2-sigma regions and shades them with different hues.
#    ax.contourf(xdata, ydata, 1-CDF, levels=[1-l for l in contour_levels], cmap=mpl.cm.summer, alpha=0.5)
#    ## Outlines 1-sigma contour
#    C1 = plt.contour(xdata, ydata, CDF, levels=[contour_levels[1]], linewidths=1, colors=['k'], alpha=0.7)
#
#                     
#
#    ax.set_xlabel(r'$E(B-V)$')
#    ax.set_ylabel(r'$R_V$')
#
#    return CDF 
#




def chi2grid(u, x, y, wave, excess, excess_var, red_law):

    CHI2 = np.zeros((len(u), len(x), len(y)))

    log( "Scanning CHI2 grid..." )
    for i, fluct in enumerate(u):   # if norm_meth = 'AVG', fluct should always be 0.  
        for j, EBV in enumerate(x):
            for k, RV in enumerate(y):
                
                ftz_curve = red_law(wave, np.zeros(wave.shape), -EBV, RV, return_excess=True)                
                                ## this is still not ideal: I would like to deal with negative fluxes before taking the log.
                nanvals = np.isnan(excess)
                nanmask = ~nanvals
                
                CHI2[i, j, k] = np.sum( (((ftz_curve - excess) + fluct)**2/excess_var)[nanmask])
    

    if np.sum(nanvals):
        print '\n\n\nWARNING. WARNGING. WARNTING.'
        print 'WARNING: THERE ARE %d BANDS WITH NAN VALUES.' % (np.sum(nanvals))
        print 'WARNING. WARNGING. WARNTING.\n\n\n'

                
    print 'len(excess)', len(excess)

    
    dof = len(excess) - 3 - (len(u) > 1)  # degrees of freedom (V-band is fixed, N_BUCKETS-1 floating data pts).  I suppose even if we use mag_avg_flux as normalization we still lose 1 dof.
    log("dof: {}".format(dof))
    log( "min CHI2: {}".format(np.min(CHI2)) )


    CHI2_dof = CHI2/dof
        
    CHI2_min = CHI2.min()
    chi2_dof_min = np.min(CHI2_dof)
    log( "min CHI2: {}".format(CHI2_min) )
    log( "min CHI2 per dof: {}".format(chi2_dof_min) )

    return CHI2, CHI2_dof, CHI2_min, chi2_dof_min




def chi2_minimization(phase, red_law, excess, excess_var, wave,
                 ebv, ebv_pad, ebv_steps, rv, rv_pad, rv_steps, u_guess, u_pad, u_steps):
    
    
    
        '''
            This is where chi2 is calculated. 
            
            I should separate the part that calculates chi2 and the part that plots.  
            
            -XH  12/6/14.
            
            
        '''
        
        print '\n\n\n Phase_index', phase, '\n\n\n'
        
               
               
        x = np.linspace(ebv-ebv_pad, ebv+ebv_pad, ebv_steps)
        y = np.linspace(rv-rv_pad, rv+rv_pad, rv_steps)
        
        X, Y = np.meshgrid(x, y)  


        ## Determine whether 2D or 3D fit.
        if u_steps > 1:
            ## 3D case
            u = np.linspace(u_guess - u_pad, u_guess + u_pad, u_steps)
        elif u_steps == 1:
            ## 2D case: only use the central value of u
            u = np.array([u_guess,])

        if ebv_steps == 1:
            x = np.array([ebv,])

        chi2_dof_min = 1e6
        chi2_rej = 400
        while chi2_dof_min > 1.5:   ## usu. the criterion is 1.5

            if len(wave) <=3:
                print 'len(wave) <= 3...breaking out of while loop...and plotting...'
                break

            chi2_dof_min_old = chi2_dof_min

    
            CHI2, CHI2_dof, CHI2_min, chi2_dof_min = chi2grid(u, x, y, wave, excess, excess_var, red_law)

            if (chi2_dof_min - chi2_dof_min_old)/chi2_dof_min_old < 1e-6:
                chi2_rej = chi2_rej*0.8


            delCHI2 = CHI2 - CHI2_min
            delCHI2_dof = CHI2_dof - chi2_dof_min

            mindex = np.where(delCHI2_dof == 0)   # Note argmin() only works well for 1D array.  -XH
            print 'mindex', mindex

            ####**** best fit values
            ## basically it's the two elements in mindex.  But each element is a one-element array; hence one needs an addition index of 0.
            mu, mx, my = mindex[0][0], mindex[1][0], mindex[2][0]
            #print 'mindex', mindex
            #print 'mu, mx, my', mu, mx, my
            best_u, best_rv, best_ebv = u[mu], y[my], x[mx]
            best_av = best_rv * best_ebv
            print 'best_u = %.5f, best_rv = %.5f, best_ebv = %.5f, best_av = %.5f' % (best_u, best_rv, best_ebv, best_av)
            

            ftz_curve = red_law(wave, np.zeros(wave.shape), -best_ebv, best_rv, return_excess=True)                
                    
            nanvals = np.isnan(excess)
            nanmask = ~nanvals
            
            pull = (((ftz_curve-excess) + best_u)/np.sqrt(excess_var))[nanmask]
            ind_chi2_terms = pull**2
            wave = wave[nanmask]

            fig = plt.figure(figsize = (20, 16))

#            ax = plt.subplot(311)
#            plt.errorbar(wave, dered_mag_norm + best_u, np.sqrt(obs_mag_var), fmt = 'r.') 
#            plt.errorbar(wave, ref_mag, np.sqrt(ref_mag_var), fmt = 'kx')
#            plt.plot(wave, np.zeros(wave.shape), 'k--')
#            plt.ylim([-0.8, 0.8])


            ax = plt.subplot(211)  
            ax.set_title('Phase: {}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
            #plt.ylim([-0.3, 0.3])
            #ax.errorbar(wave, excess - best_u - ftz_curve, np.sqrt(excess_var), fmt='r.', ms = 8, label=u'excess', alpha = 0.3) #, 's', color='black', 
            ax.plot(wave, pull, 'r.', ms = 8, label=u'pull', alpha = 0.3) #, 's', color='black', 

            ax.plot(wave, np.zeros(wave.shape), 'k--')
            
            plttext = "\n$\chi_{{min}}^2/dof = {:.2f}$" + \
                      "\n$E(B-V)={:.5f}$" + \
                      "\n$R_V={:.5f}$" + \
                      "\n$A_V={:.5f}$" + \
                      "\n$u={:.2f}$"
                      
                      

            plttext = plttext.format(chi2_dof_min, best_ebv, best_rv, best_av, best_u)

            ax.text(.95, .95, plttext, size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
            
            #ind_chi2_terms = (ftz_curve - excess + best_u)**2/excess_var
            
            ax = plt.subplot(212)   
            
            ax.plot(wave, ind_chi2_terms, 'k.') 

            # r below stands for raw strings.
            ax.text(.85, .95, r'(color excess $-$ reddening law)$^2$/$\sigma_i^2$', size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

            #plt.show()


        
            print 'chi2_rej', chi2_rej
            mask = ind_chi2_terms < chi2_rej
            #print 'mask', mask
            wave_rej = wave[~mask]
            print 'rejected wavelengths:', wave_rej

             
            wave = wave[mask]
            excess = excess[mask]
            excess_var = excess_var[mask]
    


        print 'chi2_dof_min', chi2_dof_min

        delCHI2 = CHI2 - CHI2_min
        delCHI2_dof = CHI2_dof - chi2_dof_min

        mindex = np.where(delCHI2_dof == 0)   # Note argmin() only works well for 1D array.  -XH
        print 'mindex', mindex

            
       
       
        ####**** best fit values

        ## basically it's the two elements in mindex.  But each element is a one-element array; hence one needs an addition index of 0.
        mu, mx, my = mindex[0][0], mindex[1][0], mindex[2][0]
        #print 'mindex', mindex
        #print 'mu, mx, my', mu, mx, my
        best_u, best_rv, best_ebv = u[mu], y[my], x[mx]
        print 'best_u = %.5f, best_rv = %.5f, best_ebv = %.5f ' % (best_u, best_rv, best_ebv)

        #plt.show()


        #exit(1)

#        res = minimize(chi2fun, [best_ebv, best_rv], args = (excess, excess_var, wave, best_u), tol = 1e-16, method = 'BFGS') 
#        print res.success
#        print res.x
#        print res.hess_inv
#        print 'inv Hessian:', np.sqrt(res.hess_inv)



        ####**** find 1-sigma and 2-sigma errors based on confidence


        reg_wave = np.arange(3000,10000,10)  # without regular spacing matplotlib can't plot connected curve.

        best_fit_curve = redden_fm(reg_wave, np.zeros(reg_wave.shape), -best_ebv, best_rv, return_excess=True)
        snake_hi_1sig = deepcopy(best_fit_curve)
        snake_lo_1sig = deepcopy(best_fit_curve)

        snake_hi_2sig = deepcopy(best_fit_curve)
        snake_lo_2sig = deepcopy(best_fit_curve)


        maxfluct_1sig, minfluct_1sig = best_u, best_u
        maxebv_1sig, maxebv_2sig, minebv_1sig, minebv_2sig = best_ebv, best_ebv, best_ebv, best_ebv
        maxrv_1sig, maxrv_2sig, minrv_1sig, minrv_2sig = best_rv, best_rv, best_rv, best_rv
        for i, fluct in enumerate(u):
            for e, EBV in enumerate(x):
                for r, RV in enumerate(y):
                    if delCHI2[i, e, r] < 1.0:
#                    if delCHI2_dof[i, e, r] < 1.0:
                        maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                        minebv_1sig = np.minimum(minebv_1sig, EBV)
                        maxrv_1sig = np.maximum(maxrv_1sig, RV)
                        minrv_1sig = np.minimum(minrv_1sig, RV)
                        maxfluct_1sig = np.maximum(maxfluct_1sig, fluct)
                        minfluct_1sig = np.minimum(minfluct_1sig, fluct)



                        ## for plotting uncertainty snake.
                        ## the approach is trying to find the outmost contour of all excess curves with delta_chi2 < 1
                        ## As such it finds the region where the probability of a E(V-X) curve lies within which is 68%.
                        ## This typically is a (much) smaller region than the one enclosed between the F99 curve with (RV+sig_RV, EBV+sig_EBV)
                        ## and the one with (RV-sig_RV, EBV-sig_EBV).  I think it is the correct way of representing the 1-sigma uncertainty snake. -XH

                        ## AS reverses the order of j and k; I think I'm right.  In fact if the steps are different for RV and EBV
                        ## one gets an error message if j and k are reversed.
                        redden_curve = redden_fm(reg_wave, np.zeros(reg_wave.shape), -EBV, RV, return_excess=True)  
                        snake_hi_1sig = np.maximum(snake_hi_1sig, redden_curve)  # the result is the higher values from either array is picked.
                        snake_lo_1sig = np.minimum(snake_lo_1sig, redden_curve)

                    elif delCHI2[i, e, r] < 4.0:
                    #elif delCHI2_dof[i, e, r] < 4.00:
                        maxebv_2sig = np.maximum(maxebv_2sig, EBV)
                        minebv_2sig = np.minimum(minebv_2sig, EBV)
                        maxrv_2sig = np.maximum(maxrv_2sig, RV)
                        minrv_2sig = np.minimum(minrv_2sig, RV)

                        redden_curve = redden_fm(reg_wave, np.zeros(reg_wave.shape), -EBV, RV, return_excess=True)  ## it seems that this statement should be outside the for loops.
                        snake_hi_2sig = np.maximum(snake_hi_2sig, redden_curve)  # the result is the higher values from either array is picked.
                        snake_lo_2sig = np.minimum(snake_lo_2sig, redden_curve)


        sig_hi_u = maxfluct_1sig - best_u
        sig_lo_u =  best_u - minfluct_1sig
        print 'best_u, sig_hi_u, sig_lo_u', best_u, sig_hi_u, sig_lo_u
        print 'maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig', maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig


        print 'best_ebv, best_rv', best_ebv, best_rv
        print 'maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig', maxebv_1sig, minebv_1sig, maxrv_1sig, minrv_1sig
    
        
        ebv_uncert_upper = maxebv_1sig - best_ebv
        ebv_uncert_lower = best_ebv - minebv_1sig
        rv_uncert_upper = maxrv_1sig - best_rv
        rv_uncert_lower = best_rv - minrv_1sig
        
        print 'ebv_uncert_upper, ebv_uncert_lower', ebv_uncert_upper, ebv_uncert_lower
        print 'rv_uncert_upper, rv_uncert_lower', rv_uncert_upper, rv_uncert_upper
        

        ## either choise of P seems to lead to large 1-sig error snakes -- not sure why.  Need to think about this more.  This is not a huge issue, because
        ## it's mainly a presentation issue.  The quoted RV and EBV uncertainties are not affected how P is calculated -- although the visual should agree with 
        ## the numbers.
        #P = np.sum(np.exp(-delCHI2_dof/2), axis = 0)  ## probability summed over the nuisance parameter u.  obviously un-normalized.
        P = np.sum(np.exp(-delCHI2/2), axis = 0)  ## probability summed over the nuisance parameter u.  obviously un-normalized.
        P = P/np.sum(P)  ## normalization.  This is now pmf (or less precisely, pdf).
        ##P = np.exp(-delCHI2_dof[mu, :, :]/2) ## choose the best u instead of integrating over u. There is a slight difference between contour plots based on this P vs. the P above.  But I think the P above is the correct one.
        

####*****Calculating AV uncertainty*******************


        ## Marginalized pdf -- see MIT 18.05 lecture 7 prep notes.
        Px = np.sum(P, axis = 1)
        Py = np.sum(P, axis = 0) 

        ## Note EBV_avg and RV_avg are very similar to their best-fit values.
        EBV_avg = np.sum(x*Px)
        RV_avg = np.sum(y*Py)
        ## Not sure if the line below is the right way of calculating AV averge.  I have for now adopted AV_avg = EBV_avg*RV_avg
        #AV_avg = np.sum(X*Y*P)
        AV_avg = EBV_avg*RV_avg

        print '\n\n\n Calculating AV (and RV, EBV) uncertainty...\n\n'
        ## Variance and Covariance according to MIT lecture 7

        var_x = np.sum(Px*(x-EBV_avg)**2)   
        var_y = np.sum(Py*(y-RV_avg)**2)    

        sig_ebv, sig_rv = np.sqrt(var_x), np.sqrt(var_y)
        print 'sig_ebv, sig_rv', sig_ebv, sig_rv


        ## I'm pretty sure the following way is right.  np.sum(P*np.outer(y-RV_avg, x-EBV_avg)) gives a much larger number.
        cov_xy = np.sum(P*np.outer(x-EBV_avg, y-RV_avg))  

#        print 'sigma_x, sigma_y', np.sqrt(var_x), np.sqrt(var_y)
#        print 'cov_xy', cov_xy

        best_av = best_rv*best_ebv
        var_AV = var_x*best_rv**2 + var_y*best_ebv**2 + 2*cov_xy*best_rv*best_ebv
        sig_av = np.sqrt(var_AV)
        print 'best_av, AV_avg, sig_av', best_av, AV_avg, sig_av


        #log( "\t {}".format(mindex) )
        log( "\t u={} RV={} EBV={} AV={}".format(best_u, best_rv, best_ebv, best_av) )


        ## CDF
        CDF = compute_sigma_level(P).T


## May need the following.
#        chi2dofs.append(CHI2_dof)
#        #chi2_reductions.append(CHI2_dof)
#        min_chi2s.append(chi2_dof_min)
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


        






        ####**** Calculating Hessian 
#
#        chi2_2d = -2*np.log(1-CDF)
#        chi2_2d_min = chi2_2d.min()
#        mindex2d = np.where(chi2_2d == chi2_2d_min)   # Note argmin() only works well for 1D array.  -XH
#        xind = mindex2d[0][0]
#        yind = mindex2d[1][0]
#
#
#        f = chi2_2d
#        h = x[xind+1]-x[xind]
#        k = y[yind+1]-y[yind]
#        f_xx = (f[xind+1, yind] - 2*f[xind, yind] + f[xind-1, yind])/(h**2)
#        f_yy = (f[xind, yind+1] - 2*f[xind, yind] + f[xind, yind-1])/(k**2)
#        f_xy = (f[xind+1, yind+1] - f[xind+1, yind-1] - f[xind-1, yind+1] + f[xind-1, yind-1])/(4*h*k)
#
#        print f_xx, f_yy, f_xy
#        print type(f_xx), type(f_yy), type(f_xy)
#        f_mat = np.matrix([[f_xx, f_xy], [f_xy, f_yy]])
#        print f_mat
#        H = np.linalg.inv(f_mat)
#        print "Hessian", H
#        print 'sqrt(H)', np.sqrt(H)
#        w, v = np.linalg.eig(H)
#        print 'eigen values', w
#        print 'sqrt(eigen values', np.sqrt(w)
#        exit(1)



## In terms of returned values, I don't think returning CDF is necessary for excess plot.  12/9/14

        return x, y, u, CDF, chi2_dof_min, best_u, best_ebv, best_rv, best_av, (minebv_1sig, maxebv_1sig), \
                                                 (minebv_2sig, maxebv_2sig), \
                                                 (minrv_1sig,  maxrv_1sig), \
                                                 (minrv_2sig,  maxrv_2sig), \
                                                 (sig_hi_u, sig_lo_u), \
                                                 sig_ebv, sig_rv, sig_av, \
                                                 snake_hi_1sig, snake_lo_1sig, \
                                                 snake_hi_2sig, snake_lo_2sig



def simulate_11fe(phases_12cu, obs_12cu, no_var = True, art_var = 1e-60, ebv = -1.0, rv = 2.8, del_mu = 0.0):

    '''
    simulate reddened 11fe, with its original variances removed and articial variances (initially, uniform) added.
    Also for pristine_11fe its variances are removed (at least initially).
    '''


    phases_12cu = np.array(phases_12cu)
    pristine_11fe = l.nearest_spectra(phases_12cu, l.get_11fe(no_var = no_var, loadmast=False, loadptf=False))
    art_reddened_11fe = l.nearest_spectra(phases_12cu, l.get_11fe('fm', ebv=ebv, rv=rv, art_var=art_var, loadmast=False, loadptf=False))


    phases_11fe = [t[0] for t in pristine_11fe]
    phases_reddened_11fe = [t[0] for t in art_reddened_11fe]

    print 'Fetched 11fe phases:', phases_11fe
    print 'Reddened 11fe phases:', phases_reddened_11fe

    return pristine_11fe, art_reddened_11fe

def func(wave, EBV, RV):
    return redden_fm(wave, np.zeros(wave.shape), -EBV, RV, return_excess=True) 



if __name__ == "__main__":

    '''
        
    To use artificially reddened 11fe as testing case:
    
    python photom_vs_spectral3D.py -select_SN 'red_11fe' -select_phases 0 -N_BUCKETS 1000 -del_mu 0.0 -u_guess 0.0 -u_pad 0.2 -u_steps 21 -EBV_GUESS 1.0 -EBV_PAD 0.01 -EBV_STEPS 41 -RV_GUESS 2.8 -RV_PAD 0.03 -RV_STEPS 41 -ebv_spect 1.00 -rv_spect 2.8 -art_var 5e-31 -snake 1 -unfilt
    
    
    
    To run 12cu:
    
    python photom_vs_spectral3D.py -select_SN '12cu' -select_phases 12 -N_BUCKETS 1000 -u_guess 0.0 -u_pad 0.2 -u_steps 81 -EBV_GUESS 1. -EBV_PAD 0.01 -EBV_STEPS 61 -RV_GUESS 3.0 -RV_PAD 0.03 -RV_STEPS 61 -snake 1
    
    
    Note: I can select any combination of phases I want.  E.g., I could do -select_phases 0 1 5.
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-N_BUCKETS', type = int)
    parser.add_argument('-del_mu', type = float)
    parser.add_argument('-select_SN', type = str)
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
    _ = parser.add_argument('-snake', type = int)  # just another way to add an argument to the list.


    args = parser.parse_args()
    print 'args', args
    select_SN = args.select_SN
    N_BUCKETS = args.N_BUCKETS
    del_mu = args.del_mu
    RV_GUESS = args.RV_GUESS
    RV_PAD = args.RV_PAD
    RV_STEPS = args.RV_STEPS
    EBV_GUESS = args.EBV_GUESS
    EBV_PAD = args.EBV_PAD
    EBV_STEPS = args.EBV_STEPS
    u_guess = args.u_guess
    u_pad = args.u_pad
    u_steps = args.u_steps
    select_phases = args.select_phases ## if there is only one phase select, it needs to be in the form of a 1-element array for all things to work.
    ebv_spect = args.ebv_spect
    rv_spect = args.rv_spect
    art_var = args.art_var
    snake = args.snake
    unfilt = args.unfilt

    hi_wave = 9700.
    lo_wave = 3300.
    
    ## To select all phase by enter a number greater than 11.
    if len(select_phases) > 11 or (np.array(select_phases) > 11).any():
        select_phases = range(11)

    print 'select_phases', select_phases   

    PLOTS_PER_ROW = math.ceil(len(select_phases)/2.)  # using math.ceil so that I can render the number of rows correctly for one plot.
    numrows = (len(select_phases)-1)//PLOTS_PER_ROW + 1

    print 'PLOTS_PER_ROW, numrows', PLOTS_PER_ROW, numrows

    
    ## load spectra, interpolate 11fe to 12cu phases (only first 11)
    obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:11]
    phases_12cu = [t[0] for t in obs_12cu]
    print 'Available 12cu phases', phases_12cu
 
 
    ## obs_SN is either an artificially reddened 11fe interpolated to the phases of 12cu, or 12cu itself.
    if select_SN == 'red_11fe':
        if art_var != None:
            pristine_11fe, obs_SN = simulate_11fe(phases_12cu, obs_12cu, no_var = True, art_var = art_var, ebv = -EBV_GUESS, rv = RV_GUESS, del_mu = del_mu)
        else:
            print 'To use artificially reddened 11fe, need to supply art_var.'
    elif select_SN == '12cu':
        obs_SN = obs_12cu
        pristine_11fe = l.nearest_spectra(phases_12cu, l.get_11fe(loadmast=False, loadptf=False))
        #pristine_11fe = l.interpolate_spectra(phases_12cu, l.get_11fe(loadmast=False, loadptf=False))




    ## Setting up tophat filters
    filters_bucket, zp_bucket, LOW_wave, HIGH_wave = l.generate_buckets(lo_wave, hi_wave, N_BUCKETS)  #, inverse_microns=True)
    filters_bucket = np.array(filters_bucket)
    filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff for f in filters_bucket])

    if unfilt == True:
        FEATURES = []
    else:
        FEATURES = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),\
                          (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]

    mask = filter_features(FEATURES, filter_eff_waves)
    filters_bucket = filters_bucket[mask]
    filters_bucket = filters_bucket.tolist()
   
    filter_eff_waves = filter_eff_waves[mask]

    del_wave = (HIGH_wave  - LOW_wave)/N_BUCKETS

#    print len(filters_bucket)   
#    print len(filter_eff_waves)   
#    print type(filters_bucket)   
#    print type(filter_eff_waves)   


    print 'select_SN:', select_SN
    if N_BUCKETS > 0:
        filters = filters_bucket
    else:       ## this really only applies to the case of simulated 11fe, for the purpose of checking spectral fit vs. 1000 band fit.
        if select_SN == 'red_11fe':
            ref_wave = pristine_11fe[0][1].wave
            filters = []
            mask = filter_features(FEATURES, ref_wave)   # this is only necessary so that the call to get_excess() takes the same form.
            for i, wave in enumerate(ref_wave):
                if i == np.argmin(abs(wave - V_wave)):
                    filters.append('V')
                else:
                    filters.append(i)
        else:
            print 'Spectral fit should only be used for articificially reddened 11fe.  Exiting...'
            exit(1)


    EXCESS, EXCESS_VAR, wave, V_MAG_DIFF = get_excess(phases_12cu, select_phases, filters, pristine_11fe, obs_SN, mask = mask, N_BUCKETS = N_BUCKETS, norm_meth = 'V_band')


    SN_CHISQ_DATA = []

    ## there is a nearly identical statement in plot_contour; should remove such redundancy which can easily lead to inconsistency. 
    for i, phase_index, phase in zip(range(len(select_phases)), select_phases, [phases_12cu[i] for i in select_phases]):  ## there is a nearly 
        
        

        ## plot_contour3D() is where chi2 is calculated. 
        x, y, u, CDF, chi2_dof_min, \
        best_u, best_ebv, best_rv, best_av, \
            ebv_1sig, ebv_2sig, \
            rv_1sig, rv_2sig, \
            sig_u, \
            sig_ebv, sig_rv, sig_av, \
            snake_hi_1sig, snake_lo_1sig, \
            snake_hi_2sig, snake_lo_2sig = chi2_minimization(phase, redden_fm, EXCESS[phase_index], EXCESS_VAR[phase_index],
                                            wave, EBV_GUESS, EBV_PAD, EBV_STEPS, RV_GUESS, RV_PAD, RV_STEPS, u_guess, u_pad, u_steps)
        
        ## I should do minus best_u below because: in fitting I do (excess - u - FTZ).  But excess = V_mag_diff - X_mag_diff.  Thus del_V should be lowered by u.
        ## See eq 1 and 2 in draft.   12/22/14
        del_mu = V_MAG_DIFF[phase_index] - best_u - best_av
        sig_del_mu = np.sqrt((0.5*sig_u[0] + 0.5*sig_u[1])**2 + sig_av**2)   ## ------------> this is strictly speaking not kosher: there must be a covariance between u and AV.
                                                     ## Though I'm hoping this won't affect things too much.  Should correct this.  Or get distance modulus
                                                     ## another way. <------------------   Dec 22, 2014
                                                    

#        print 'i, phase_index, phase', i, phase_index, phase
#        exit(1)
        SN_CHISQ_DATA.append({'phase'       : phase,
                                 'WAVE'         : wave,
                                 'FILTERS'      : filters,
                                 'EXCESS'       : EXCESS[phase_index],
                                 'EXCESS_VAR'   : EXCESS_VAR[phase_index],
                                 'x'            : x,
                                 'y'            : y,
                                 'u'            : u,
                                 'CDF'          : CDF,
                                 'CHI2_DOF_MIN' : chi2_dof_min,
                                 'BEST_EBV'     : best_ebv,
                                 'BEST_RV'      : best_rv,
                                 'BEST_u'       : best_u,
                                 'BEST_AV'      : best_av,
                                 'EBV_1SIG'     : ebv_1sig,
                                 'EBV_2SIG'     : ebv_2sig,
                                 'RV_1SIG'      : rv_1sig,
                                 'RV_2SIG'      : rv_2sig,
                                 'hi_1sig'      : snake_hi_1sig,
                                 'lo_1sig'      : snake_lo_1sig,
                                 'hi_2sig'      : snake_hi_2sig,
                                 'lo_2sig'      : snake_lo_2sig,
                                 'SIG_U'        : sig_u,
                                 'SIG_EBV'      : sig_ebv,
                                 'SIG_RV'       : sig_rv,
                                 'SIG_AV'       : sig_av,
                                 'DEL_MU'       : del_mu,
                                 'SIG_DEL_MU'   : sig_del_mu
                                 })



    SNdata = select_SN + '_CHISQ_DATA' + '_' + str(len(select_phases))

    if unfilt:
        filenm = SNdata + '_unfilt.p'
    else:
        filenm = SNdata + '_filtered.p'


    pickle.dump(SN_CHISQ_DATA, open(filenm, 'wb'))


    SN_CHISQ_DATA_out = pickle.load(open(filenm, 'rb'))   


    plots.plot_contours(select_SN, SN_CHISQ_DATA_out, unfilt)

    if len(select_phases) == 11:
        plots.plot_summary(select_SN, SN_CHISQ_DATA_out, unfilt)
                            
    plots.plot_phase_excesses(select_SN, SN_CHISQ_DATA, redden_fm, unfilt, snake = snake, FEATURES = FEATURES)


    ## Using scipy.curve_fit()
    excess_phase = EXCESS[select_phases[0]]
    excess_sig = np.sqrt(EXCESS_VAR[select_phases[0]]) 
    popt, pcov = curve_fit(func, wave, excess_phase, p0 = [1.0, 2.8], sigma = excess_sig, absolute_sigma = True)

    print 'optimal m and b', popt
    print '1-sigma uncertainties on EBV and RV', np.sqrt(np.diag(pcov))
    print 'covariance matrix between EBV and RV', pcov


    plt.show()

########################################## Function Junk Yard (I no longer use these but am keeping them for the time being ##################################





## I don't use the following approach to calculate the error snakes -- they seem to be too fat.  So this function at the moment is defunct.  12/17/14.
#def plot_snake(ax, wave, best_fit_curve, red_law, x, y, CDF, plot2sig=False):
#    
#    '''
#    Date thrown into junk yard: 12/17/14
#    
#    Reason: The approach used here to calculate the error snakes is based on CDF (see the last code segement), which sounds right but the error snakes produced seem to be too fat for artificially reddened 11fe.  I need to think this through (why is it like this) before throwing away this function.   12/17/14.
#
#    '''
#    
#    snake_hi_1sig = deepcopy(best_fit_curve)
#    snake_lo_1sig = deepcopy(best_fit_curve)
#    if plot2sig:
#        snake_hi_2sig = deepcopy(best_fit_curve)
#        snake_lo_2sig = deepcopy(best_fit_curve)
#
#    for i, EBV in enumerate(x):
#        for j, RV in enumerate(y):
#            if CDF[j,i]<0.683:
#                red_curve = red_law(wave, np.zeros(wave.shape), -EBV, RV, return_excess=True)
#                snake_hi_1sig = np.maximum(snake_hi_1sig, red_curve)
#                snake_lo_1sig = np.minimum(snake_lo_1sig, red_curve)
#            elif plot2sig and CDF[j,i]<0.955:
#                red_curve = red_law(wave, np.zeros(wave.shape), -EBV, RV, return_excess=True)
#                snake_hi_2sig = np.maximum(snake_hi_2sig, red_curve)
#                snake_lo_2sig = np.minimum(snake_lo_2sig, red_curve)
#    
#    ax.fill_between(wave, snake_lo_1sig, snake_hi_1sig, facecolor='black', alpha=0.3)
#    if plot2sig:
#        ax.fill_between(wave, snake_lo_2sig, snake_hi_2sig, facecolor='black', alpha=0.1)
#    
#    return interp1d(wave, snake_lo_1sig), interp1d(wave, snake_hi_1sig)
#
#
### Note this function doesn't seem to be in active use.  May put it in the junkyard.  1/11/15
#def chi2fun(params, excess = 0, excess_var = 1e-60, wave = np.linspace(3000, 10000, 1000), fluct = 0):
#    
#
#
#    ebv = params[0]
#    rv = params[1]
#                
#    ftz_curve = redden_fm(wave, np.zeros(wave.shape), -ebv, rv, return_excess=True)
#                    
##    nanvals = np.isnan(excess)
##    nanmask = ~nanvals
#
#
#    chi2 = np.sum( (((ftz_curve-excess) + fluct)**2/excess_var) )   #[nanmask])
#    
#
##    if np.sum(nanvals):
##        print '\n\n\nWARNING. WARNGING. WARNTING.'
##        print 'WARNING: THERE ARE %d BANDS WITH NAN VALUES.' % (np.sum(nanvals))
##        print 'WARNING. WARNGING. WARNTING.\n\n\n'
#
#                
#    dof = len(wave) - 3 #- (len(u) > 1)  # degrees of freedom (V-band is fixed, N_BUCKETS-1 floating data pts).  I suppose even if we use mag_avg_flux as normalization we still lose 1 dof.
#
#
#    chi2_dof = chi2/dof
#
#    return chi2_dof
