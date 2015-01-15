'''
Initial Date: 12/23/14

A bunch of plotting routines

Need to improve: there is redundancy in how information is read from SN12CU_CHISQ_DATA in plot_contours() and plot_excess().

'''

import argparse
from copy import deepcopy
import math
import pickle


import loader as l
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import *
#Formatter


import sncosmo as snc

from itertools import izip
from loader import redden_fm, redden_pl, redden_pl2
from pprint import pprint
from scipy.stats import chisquare
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from mag_spectrum_fitting import ABmag_nu, extract_wave_flux_var, flux2mag, log, filter_features

#from matplotlib import rc



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



def plot_contours(SNname, SN12CU_CHISQ_DATA, unfilt):


## , best_av, sig_av  
####***** Plotting confidence contours **************************


    phases = np.array([d['phase'] for d in SN12CU_CHISQ_DATA])

    PLOTS_PER_ROW = math.ceil(len(SN12CU_CHISQ_DATA)/2.)


#np.array([phase_index for phase_index in SN12CU_CHISQ_DATA['phase']])
    print 'phases:', phases
   
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1

    x = SN12CU_CHISQ_DATA[0]['x']
    y = SN12CU_CHISQ_DATA[0]['y']
    
    X, Y = np.meshgrid(x, y)  
 
    contour_levels = [0.0, 0.683, 0.955, 1.0]

    fig = plt.figure(figsize = (24, 15))


    for phase_index, phase, d in zip(range(len(SN12CU_CHISQ_DATA)), phases, SN12CU_CHISQ_DATA):

    ## Need to think about how to do contour plots -- basically what Zach and I went through in Starbucks in October.
    ## Don't delete below yet.  There is useful code below about the plotting styles. 

        ## plot contours


        ax = plt.subplot(numrows, PLOTS_PER_ROW, phase_index+1)

        best_ebv = d['BEST_EBV'] 
        minebv_1sig = d['EBV_1SIG'][0] 
        maxebv_1sig = d['EBV_1SIG'][1]

        best_rv = d['BEST_RV'] 
        minrv_1sig = d['RV_1SIG'][0] 
        maxrv_1sig = d['RV_1SIG'][1]
        
        CDF = d['CDF']
        best_av = d['BEST_AV']
        sig_av = d['SIG_AV']


        best_av = d['BEST_AV']
        sig_av = d['SIG_AV']


        
        ## plots 1- and 2-sigma regions and shades them with different hues.
        plt.contourf(X, Y, 1-CDF, levels=[1-l for l in contour_levels], cmap=mpl.cm.summer, alpha=0.5)

        ## Outlines 1-sigma contour  # this apparently is not used.  Can be deleted soon.  1/11/15
        # C1 = plt.contour(X, Y, CDF, levels=[contour_levels[1]], linewidths=1, colors=['k'], alpha=0.7)
        
        #plt.contour(X, Y, CHISQ-chisq_min, levels=[1.0, 4.0], colors=['r', 'g'])
        
        ## mark minimum
        plt.scatter(best_ebv, best_rv, marker='s', facecolors='r')
        
        # show results on plot
        if phase_index%6==0:
                plttext1 = "Phase: {}".format(phase)
        else:
                plttext1 = "{}".format(phase)
                
        plttext2 = "$E(B-V)={:.5f}^{{{:+.5f}}}_{{{:+.5f}}}$" + \
                   "\n$R_V={:.5f}^{{{:+.5f}}}_{{{:+.5f}}}$" + \
                   "\n$A_V={:.5f}^{{{:+.5f}}}_{{{:+.5f}}}$"
        
        plttext2 = plttext2.format(best_ebv, maxebv_1sig-best_ebv, minebv_1sig-best_ebv,
                                   best_rv, maxrv_1sig-best_rv, minrv_1sig-best_rv,
                                   best_av, sig_av, -sig_av
                                   )
                                           
        if phase not in [14.5, 16.5, 18.5, 21.5, 23.5]:
                ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                ax.text(.04, .85, plttext2, size=INPLOT_LEGEND_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                #ax.axhspan(2.9, (y.max()), facecolor='k', alpha=0.1)  # this produces a gray band, which I'm not using.  12/23/14
        
        else:
                ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                ax.text(.04, .32, plttext2, size=INPLOT_LEGEND_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                #ax.axhspan(3.32, (y.max()), facecolor='k', alpha=0.1)
                #ax.axhspan((y.min()), 2.5, facecolor='k', alpha=0.1)
                
                
        # format subplot...
        plt.ylim(y.min(), y.max())
        plt.xlim(x.min(), x.max())
        
        ax.set_yticklabels([])
        ax2 = ax.twinx()
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(y.min(), y.max())
        
        if phase_index%6 == 5:
                ax2.set_ylabel('\n$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=5)
        if phase_index%6 == 0:
                ax.set_ylabel('$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=-2)
        if phase_index>=5:
                ax.set_xlabel('\n$E(B-V)$', fontsize=AXIS_LABEL_FONTSIZE)
        
        ## format x labels
        ax.locator_params(nbins=6)  # this sets the number of tickmarks.
        labels = ax.get_xticks().tolist()
        
        # get rid of the first (0) and the last label (1.4); i.e. the labels at the ends.
        labels[0] = labels[-1] = ''   

        #exit(1)
        ax.set_xticklabels(labels)
        ax.get_xaxis().set_tick_params(direction='in', pad=-20)
        
        # format y labels
        labels = ax2.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax2.set_yticklabels(labels)
        ax2.get_yaxis().set_tick_params(direction='in', pad=-30)
        
        plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
        plt.setp(ax2.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)



    fig.subplots_adjust(left=0.04, bottom=0.08, right=0.95, top=0.92, hspace=.06, wspace=.1)
    #fig.suptitle('SN2012CU: $E(B-V)$ vs. $R_V$ Contour Plot per Phase', fontsize=TITLE_FONTSIZE)


    if len(phases) > 1:
        SNcontour = SNname + '_contour'+ '_' + str(len(phases))
    else:
        SNcontour = SNname + '_contour'+ '_' + 'phase' + str(phases[0])
    
    
    if unfilt:
        filenm = SNcontour + '_unfilt.png'
    else: 
        filenm = SNcontour + '_filtered.png'

    plt.savefig(filenm)

    return


#def plot_phase_excesses(SN12CU_CHISQ_DATA, redden_fm, snake = snake):

def plot_phase_excesses(SNname, SN12CU_CHISQ_DATA, red_law, unfilt, snake = True, FEATURES = []):

    ''' 
     
    
    '''
    wave = SN12CU_CHISQ_DATA[0]['WAVE']
    phases = np.array([d['phase'] for d in SN12CU_CHISQ_DATA])


    if len(phases) < 6:
        numrows = 1
        PLOTS_PER_ROW = len(phases)
    else:
        PLOTS_PER_ROW = math.ceil(len(SN12CU_CHISQ_DATA)/2.)
        numrows = (len(phases)-1)//PLOTS_PER_ROW + 1

    
    print "Plotting excesses of", SNname, " with best fit from contour..."
    
    if len(phases) > 6:
        fig = plt.figure(figsize = (24, 15))
    else:
        fig = plt.figure(figsize = (20, 14))
    
    #numrows = (len(EXCESS)-1)//PLOTS_PER_ROW + 1
    ## Keep; may need this later: pmin, pmax = np.min(phases), np.max(phases)
    
    ## may need this for running all phases    for i, d, sn11fe_phase in izip(xrange(len(SN12CU_CHISQ_DATA)), SN12CU_CHISQ_DATA, sn11fe):
    ##for i, phase, d in zip(range(len(SN12CU_CHISQ_DATA)), phases, SN12CU_CHISQ_DATA):
    for phase_index, phase, d in zip(range(len(phases)), phases, SN12CU_CHISQ_DATA):



        print "Plotting phase {} ...".format(phase)
        print 'phase_index', phase_index            
        
        if len(phases) > 1:
            ax = plt.subplot(numrows, PLOTS_PER_ROW, phase_index + 1)   
        else: 
            ax = plt.subplot(211)   

        phase_excess = d['EXCESS']   #[phase_index][j] for j, f in enumerate(filters)])
        phase_excess_var = d['EXCESS_VAR']  #  np.array([EXCESS_VAR[phase_index][j] for j, f in enumerate(filters)])



        ## Keep the following two line in case I want to plot the symbols with different colors for different phases.  12/10/14
        #mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))
        #plt.plot(filter_eff_waves, phase_excesses, 's', color=mfc_color, ms=8, mec='none', mfc=mfc_color, alpha=0.8)

        reference_curve = red_law(wave, np.zeros(wave.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)

        plt.errorbar(wave, phase_excess - d['BEST_u'] - reference_curve, np.sqrt(phase_excess_var), fmt='r.', ms = 8, label=u'excess', alpha = 0.3) #, 's', color='black', ms=8) #, mec='none', mfc=mfc_color, 
        plt.plot(wave, np.zeros(wave.shape), 'k--')
 
        ## plot best-fit reddening curve and uncertainty snake

## Temporarily blocked.  1/8/15
#        reg_wave = np.arange(3000,10000,10)
#        #xinv = 10000./x # can probably delete this soon.  12/18/14
#        red_curve = red_law(reg_wave, np.zeros(reg_wave.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)
#        plt.plot(reg_wave, red_curve, 'k--')
#
#
#
#        if snake:
#            ax.fill_between(reg_wave, d['lo_1sig'], d['hi_1sig'], facecolor='black', alpha=0.5)
#            ax.fill_between(reg_wave, d['lo_2sig'], d['hi_2sig'], facecolor='black', alpha=0.3)
#
#
#
#        ## plot where V band is.   -XH
#        plt.plot([reg_wave.min(), reg_wave.max()], [0, 0] ,'--')
#        plt.plot([V_wave, V_wave], [red_curve.min(), red_curve.max()] ,'--')
        
    

        ## Not sure what the following is for.  If I don't find use for it get rid of it.  12/18/14
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
            
        if snake:
            plttext = "\n$\chi_{{min}}^2/dof = {:.2f}$" + \
                      "\n$E(B-V)={:.5f}^{{{:+.5f}}}_{{{:+.5f}}}$" + \
                      "\n$R_V={:.5f}^{{{:+.5f}}}_{{{:+.5f}}}$" + \
                      "\n$A_V={:.5f}^{{{:+.5f}}}_{{{:+.5f}}}$" + \
                      "\n$u={:.2f}^{{{:+.2f}}}_{{{:+.2f}}}$"
                      
                      

            plttext = plttext.format(d['CHI2_DOF_MIN'],
                                     d['BEST_EBV'], d['EBV_1SIG'][1]-d['BEST_EBV'], d['EBV_1SIG'][0]-d['BEST_EBV'],
                                     d['BEST_RV'], d['RV_1SIG'][1]-d['BEST_RV'], d['RV_1SIG'][0]-d['BEST_RV'],
                                     d['BEST_AV'], d['SIG_AV'], -d['SIG_AV'],
                                     d['BEST_u'], d['SIG_U'][1], -d['SIG_U'][0])
        else:
            plttext = "\n$\chi_{{min}}^2/dof = {:.2f}$" + \
                      "\n$E(B-V)={:.5f}$" + \
                      "\n$R_V={:.5f}$" + \
                      "\n$A_V={:.5f}$" + \
                      "\n$u={:.2f}$"
                      
                      

            plttext = plttext.format(d['CHI2_DOF_MIN'],
                                     d['BEST_EBV'],
                                     d['BEST_RV'],
                                     d['BEST_AV'],
                                     d['BEST_u'])



        # r below stands for raw strings.
        ax.text(.85, .95, r'color excess $-$ reddening law', size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)


        ax.text(.95, .45, plttext, size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
   


#        ## Alternative way of labeling phases -- with text, rather than subplot title.
#        if phase_index%6==0:
#                plttext1 = "Phase: {}".format(phase)
#        else:
#                plttext1 = "{}".format(phase)
#                
#                                           
#        ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
#                horizontalalignment='left',
#                verticalalignment='top',
#                transform=ax.transAxes)
        

   
        ## format subplot
        if phase_index%PLOTS_PER_ROW == 0:
            ax.set_title('{0} - Phase: {1}'.format(SNname, phase), fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel('$E(V-X)$', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_title('{}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
    
        plt.xlim(3000, 10000)
        plt.ylim(-3.0, 2.0)
        
        labels = ax.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_yticklabels(labels)

## temporarily blocked 1/8/15
#        ax.locator_params(axis = 'x', nbins=4)  # this sets the number of tickmarks at 5.
#        labels = ax.get_xticks().tolist()
#        labels = [int(label) for label in labels]
#        labels[0] = labels[-1] = '' # This is to remove the xticks at the the two end points.
#        #major_formatter = FormatStrFormatter('%4.0f') ## These two statements don't seem to do anything.
#        #ax.xaxis.set_major_formatter(major_formatter)
#        ax.set_xticklabels(labels)
        
        if phase_index>=5:
            #fig.gca().set_xlabel(r'wavelength $5000 \AA$')
            ax.set_xlabel(r'$\lambda (\AA)$', fontsize=AXIS_LABEL_FONTSIZE)


        plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
        plt.setp(ax.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)

        ## Trimming margins so that more space is used for the plots, and controls space between subplots.
        fig.subplots_adjust(left=0.04, bottom=0.08, right=0.95, top=0.92, hspace=.12, wspace=.16)



        ## plot each term in chi2
        if len(phases) == 1:
            ftz_curve = red_law(wave, np.zeros(wave.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)                
                
            
            ind_chi2_terms = (ftz_curve - phase_excess + d['BEST_u'])**2/phase_excess_var
            
            ax = plt.subplot(212)   
            
            ax.plot(wave, ind_chi2_terms, 'k.') 

            # r below stands for raw strings.
            ax.text(.85, .95, r'(color excess $-$ reddening law)$^2$/$\sigma_i^2$', size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)



            ## plot derivative spectrum
            
            obs_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:11]
            phases_12cu = [t[0] for t in obs_12cu]
            pristine_11fe = l.nearest_spectra(phases_12cu, l.get_11fe(loadmast=False, loadptf=False))

            obs = obs_12cu[0]
            ref = pristine_11fe[0]

            obs_wave = obs[1].wave
            obs_flux = obs[1].flux
            dereddened = redden_fm(obs_wave, obs_flux, 1.0, 2.84, return_excess = False)

            #SN = snc.Spectrum(ref[1].wave, redden_fm(ref[1].wave, ref[1].flux, ebv, rv), SN_obs[1].error)

            ref_wave = ref[1].wave
            ref_flux = ref[1].flux
            varz = ref[1].error  # it's called error because that's one of the attributes defined in sncosmo
            weights = 1/varz
            
            del_v = 6000
            c = 3e5
            frac_wdow = del_v/c
            
            cut_wave = ref_wave[-1]*frac_wdow
            idx = ref_wave < ref_wave[-1] - ref_wave[-1]*frac_wdow
            wave_cut = ref_wave[idx]
            w_ctr = []
            F_log =[]
            dF_log = []
            #wv = 5000.
            for i, wv in enumerate(wave_cut):
                window = frac_wdow*wv
                idx = (ref_wave < wv + window) * (ref_wave > wv)
                wvs = ref_wave[idx]
                if wvs.size == 0:
                    print '0 size wavelenght array at', i, wv
                    raise KeyboardInterrupt
                wv0 = wvs.mean()
                #print 'wv0', wv0
                wvs = wvs - wv0
                f = ref_flux[idx]
                wgt = weights[idx]
                
                p = np.polyfit(wvs, f, 3, w=wgt)   
                #print 'p', p
                #sg = np.poly1d(p)
                
                w_ctr.append(wv0)
                F_log.append(np.log10(p[3]))
                dF_log.append(p[2]/(p[3]*np.log(10)))

            w_ctr = np.array(w_ctr)
            F_log = np.array(F_log)
            dF_log = np.array(dF_log)

            #print w_ctr
            #print F_log


            #ax.plot(w_ctr, (F_log+13.3)*1.5e-5, '-')  #, wave, np.log10(flux), '-')

            mask = filter_features(FEATURES, w_ctr)
            ax.plot(w_ctr[mask], (dF_log[mask]**2)*1e8, 'x')
            plt.ylim([0, 700])
            ax.text(.85, .85, 'x: derivative^2 after SG smoothing', color = 'b', size=INPLOT_LEGEND_FONTSIZE, \
                    horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

            #show()
            #xlim([30



    if len(phases) > 1:
        SNexcess = SNname + '_excess'+ '_' + str(len(phases))
    else:
        SNexcess = SNname + '_excess'+ '_' + 'phase' + str(phases[0])



    if unfilt:
        filenm = SNexcess + '_unfilt.png'
    else: 
        filenm = SNexcess + '_filtered.png'

    plt.savefig(filenm)
            


    return # plot_excess()





def plot_summary(SNname, SN12CU_CHISQ_DATA, unfilt):

    ''' 
     
    
    '''

    
    print "Plotting time dependence", SNname


    phases = np.array([d['phase'] for d in SN12CU_CHISQ_DATA])


#np.array([phases for phases in SN12CU_CHISQ_DATA['phase']])
    print phases

   
    ebvs = np.array([d['BEST_EBV'] for d in SN12CU_CHISQ_DATA])
    sig_lo_ebv = np.array([d['BEST_EBV'] - d['EBV_1SIG'][0] for d in SN12CU_CHISQ_DATA])
    sig_hi_ebv = np.array([d['EBV_1SIG'][1] - d['BEST_EBV'] for d in SN12CU_CHISQ_DATA])
    sig_ebv = np.array([d['SIG_EBV'] for d in SN12CU_CHISQ_DATA])



    rvs = np.array([d['BEST_RV'] for d in SN12CU_CHISQ_DATA])
    sig_lo_rv = np.array([d['BEST_RV'] - d['RV_1SIG'][0] for d in SN12CU_CHISQ_DATA])
    sig_hi_rv = np.array([d['RV_1SIG'][1] - d['BEST_RV'] for d in SN12CU_CHISQ_DATA])
    sig_rv = np.array([d['SIG_RV'] for d in SN12CU_CHISQ_DATA])


    avs = np.array([d['BEST_AV'] for d in SN12CU_CHISQ_DATA])
    sig_av = np.array([d['SIG_AV'] for d in SN12CU_CHISQ_DATA])
    
    
    var_intrins_cablib = 0.02**2 + 0.03**2 ## ---------> strictly speaking, should convert 0.02 fractional flux error into mag error.  
                                           ## for 11fe, the calib error is more like 0.03.  Note that these are systemtic (correlated) uncertainties in wavelength
                                           ## (and so they don't average out over wavelengths),
                                           ## but they are NOT correlated across phases -- thus adding them in quadrature to the uncertainties from fitting phase by phase is
                                           ## is the right thing to do.  That is, if one observes more phases, these calibration uncertainties do "avearge out".
                                           ## Andrew's loader actually returns the calib uncertainties, and for 11fe they are different for different phases and can be as 
                                           ## high as 5% (if I remember correctly). So should do it right next time.  Though it's still subdominant to all other errors. <--- 12/22/14
    del_mus = np.array([d['DEL_MU'] for d in SN12CU_CHISQ_DATA])
    sig_del_mu = np.array([np.sqrt(d['SIG_DEL_MU']**2 + var_intrins_cablib) for d in SN12CU_CHISQ_DATA])
    
    print 'del_mu, sig_del_mu', del_mus, sig_del_mu


####***** Calculating Overall Average ************


    EBV_overall_avg = np.average(ebvs, weights = 1/sig_ebv**2) 
    RV_overall_avg = np.average(rvs, weights = 1/sig_rv**2) 
    AV_overall_avg = np.average(avs, weights = 1/sig_av**2)    
    del_mu_overall_avg = np.average(del_mus, weights = 1/sig_del_mu**2) 


    N = len(sig_ebv)
    sig_EBV_overall = np.sqrt(np.sum(sig_ebv**2)/N)
    sig_RV_overall = np.sqrt(np.sum(sig_rv**2)/N)
    sig_AV_overall = np.sqrt(np.sum(sig_av**2)/N)    
    sig_del_mu_overall = np.sqrt(np.sum(sig_del_mu**2)/N)


#    phases_early = phases[:2]
    AV_early_avg = np.average(avs[:2], weights = 1/sig_av[:2]**2)
    sig_AV_early = np.sqrt(np.sum(sig_av[:2]**2)/len(sig_av[:2]))

#    phases_late = phases[2:]
    AV_late_avg = np.average(avs[2:], weights = 1/sig_av[2:]**2)
    sig_AV_late = np.sqrt(np.sum(sig_av[2:]**2)/len(sig_av[2:]))





    print 'EBV_ovarall_avg, sig_EBV_overall', EBV_overall_avg, sig_EBV_overall   
    print 'RV_ovarall_avg, sig_RV_overall', RV_overall_avg, sig_RV_overall   
    print 'AV_ovarall_avg, sig_AV_overall', AV_overall_avg, sig_AV_overall   
    print 'del_mu_ovarall_avg, sig_del_mu_overall', del_mu_overall_avg, sig_del_mu_overall   


####***** Plotting Time Dependence ***************


    fig = plt.figure(figsize = (15, 12))
    ax = fig.add_subplot(411) 
    plt.errorbar(phases, ebvs, yerr = [sig_lo_ebv, sig_hi_ebv], fmt = 'b.', ms = 8)
    plt.plot(phases, EBV_overall_avg*np.ones(phases.shape), 'k--')
    ax.fill_between(phases, (EBV_overall_avg+sig_EBV_overall)*np.ones(phases.shape), (EBV_overall_avg-sig_EBV_overall)*np.ones(phases.shape), \
                       facecolor='black', alpha=0.1)
    ebv_txt = "$E(B-V) = {:.3f}\pm{:.3f}$".format(EBV_overall_avg, sig_EBV_overall)
    ax.text(.04, .9, ebv_txt, size=AXIS_LABEL_FONTSIZE,\
                            horizontalalignment='left', \
                            verticalalignment='top', \
                            transform=ax.transAxes)
                            
    plt.ylabel('$E(B-V)$', fontsize=AXIS_LABEL_FONTSIZE)   
    ax.locator_params(axis = 'y', nbins=6)  # this sets the number of tickmarks.
    labels = ax.get_yticks().tolist()
    ax.set_yticklabels(labels)



    ax = fig.add_subplot(412) 
    plt.errorbar(phases, rvs, yerr = [sig_lo_rv, sig_hi_rv], fmt = 'r.', ms = 8)
    plt.plot(phases, RV_overall_avg*np.ones(phases.shape), 'k--')
    ax.fill_between(phases, (RV_overall_avg+sig_RV_overall)*np.ones(phases.shape), (RV_overall_avg-sig_RV_overall)*np.ones(phases.shape), \
                       facecolor='black', alpha=0.1)
    rv_txt = "$R_V = {:.3f}\pm{:.3f}$".format(RV_overall_avg, sig_RV_overall)
    ax.text(.04, .9, rv_txt, size=AXIS_LABEL_FONTSIZE,\
                            horizontalalignment='left', \
                            verticalalignment='top', \
                            transform=ax.transAxes)

    plt.ylabel('$R_V$', fontsize=AXIS_LABEL_FONTSIZE)
    ax.locator_params(axis = 'y', nbins=6)  # this sets the number of tickmarks at 5.
    labels = ax.get_yticks().tolist()
    ax.set_yticklabels(labels)





    ax = fig.add_subplot(413) 
    plt.errorbar(phases, avs, yerr = sig_av, fmt = 'g.', ms = 8)
    
    plt.plot(phases, AV_overall_avg*np.ones(phases.shape), 'k--')
    ax.fill_between(phases, (AV_overall_avg+sig_AV_overall)*np.ones(phases.shape), (AV_overall_avg-sig_AV_overall)*np.ones(phases.shape), \
                       facecolor='black', alpha=0.3)
                       
    plt.plot(phases, AV_early_avg*np.ones(phases.shape), 'k:')
    ax.fill_between(phases, (AV_early_avg+sig_AV_early)*np.ones(phases.shape), (AV_early_avg-sig_AV_early)*np.ones(phases.shape), \
                       facecolor='cyan', alpha=0.05)

    plt.plot(phases, AV_late_avg*np.ones(phases.shape), 'k:')
    ax.fill_between(phases, (AV_late_avg+sig_AV_late)*np.ones(phases.shape), (AV_late_avg-sig_AV_late)*np.ones(phases.shape), \
                       facecolor='cyan', alpha=0.05)




    av_txt = "$A_V = {:.3f}\pm{:.3f}$".format(AV_overall_avg, sig_AV_overall)
    ax.text(.04, .9, av_txt, size=AXIS_LABEL_FONTSIZE, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    av_txt = "$A_V^{early} = %.2f \pm %.2f$" % (AV_early_avg, sig_AV_early) #+ "$\pm$"
    ax.text(.4, .15, av_txt, size=16, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    av_txt = "$A_V^{late} = %.2f \pm %.2f$" % (AV_late_avg, sig_AV_late) #+ "$\pm$"
    ax.text(.4, .85, av_txt, size=16, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)




    plt.ylabel('$A_V$', fontsize=AXIS_LABEL_FONTSIZE)
    ax.locator_params(axis = 'y', nbins=6)  # this sets the number of tickmarks at 5.
    labels = ax.get_yticks().tolist()
    ax.set_yticklabels(labels)

    
    ax = fig.add_subplot(414) 
    plt.errorbar(phases, del_mus, yerr = sig_del_mu, fmt = 'm.', ms = 8)
    plt.plot(phases, del_mu_overall_avg*np.ones(phases.shape), 'k--')
    ax.fill_between(phases, (del_mu_overall_avg+sig_del_mu_overall)*np.ones(phases.shape), (del_mu_overall_avg-sig_del_mu_overall)*np.ones(phases.shape), \
                       facecolor='black', alpha=0.1)
    del_mu_txt = "$\Delta\mu = {:.3f}\pm{:.3f}$".format(del_mu_overall_avg, sig_del_mu_overall)
    ax.text(.04, .9, del_mu_txt, size=AXIS_LABEL_FONTSIZE,\
                            horizontalalignment='left', \
                            verticalalignment='top', \
                            transform=ax.transAxes)

    plt.ylabel('$\Delta\mu$', fontsize=AXIS_LABEL_FONTSIZE)
    ax.locator_params(axis = 'y', nbins=6)  # this sets the number of tickmarks at 5.
    labels = ax.get_yticks().tolist()
    ax.set_yticklabels(labels)

    plt.xlabel('Phases', fontsize=AXIS_LABEL_FONTSIZE)
    
    plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
    plt.setp(ax.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)

    
    
    SNsummary = SNname + '_summary' + '_' + str(len(phases))
    
    if unfilt:
        filenm = SNsummary + '_unfilt.png'
    else: 
        filenm = SNsummary + '_filtered.png'

    plt.savefig(filenm)



                            

#    for i, phase_index, phase, d in zip(range(len(SN12CU_CHISQ_DATA)), select_phases, [phases[i] for i in select_phases], SN12CU_CHISQ_DATA):
       



if __name__ == "__main__":
    

    '''
        python plots.py -SNname '12cu' -num_phases 11 -unfilt -snake 1
        
    '''
  
    parser = argparse.ArgumentParser()
    parser.add_argument('-SNname', type = str)
    parser.add_argument('-num_phases', type = int) 
    parser.add_argument('-snake', type = int)
    parser.add_argument('-unfilt', '--unfilt', action='store_true')  # just another way to add an argument to the list.


    args = parser.parse_args()
    print 'args', args
    SNname = args.SNname
    num_phases = args.num_phases
    snake = args.snake
    unfilt = args.unfilt
  
#    SNname = 'SN12CU'
#    num_phases = 3
#    unfilt = 1
    if unfilt == True:
        FEATURES = []
    else:
        FEATURES = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),\
                          (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]



    SNdata = SNname + '_CHISQ_DATA' + '_' + str(num_phases)

    if unfilt:
        filenm = SNdata + '_unfilt.p'
    else:
        filenm = SNdata + '_filtered.p'
    
    ## Should change this to a generic name - 1/13/15
    SN12CU_CHISQ_DATA = pickle.load(open(filenm, 'rb'))   

    print 'filenm', filenm
    print 'len(SN12CU_CHISQ_DATA)', len(SN12CU_CHISQ_DATA)


    plot_contours(SNname, SN12CU_CHISQ_DATA, unfilt)
    plot_phase_excesses(SNname, SN12CU_CHISQ_DATA, redden_fm, unfilt, snake = snake, FEATURES = FEATURES)
    if num_phases > 1:
        plot_summary(SNname, SN12CU_CHISQ_DATA, unfilt)
