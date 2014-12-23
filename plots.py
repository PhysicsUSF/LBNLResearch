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


    for i, phase, d in zip(range(len(SN12CU_CHISQ_DATA)), phases, SN12CU_CHISQ_DATA):

    ## Need to think about how to do contour plots -- basically what Zach and I went through in Starbucks in October.
    ## Don't delete below yet.  There is useful code below about the plotting styles. 

        ## plot contours


        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)

        best_ebv = d['BEST_EBV'] 
        minebv_1sig = d['EBV_1SIG'][0] 
        maxebv_1sig = d['EBV_1SIG'][0]

        best_rv = d['BEST_RV'] 
        minrv_1sig = d['RV_1SIG'][0] 
        maxrv_1sig = d['RV_1SIG'][0]
        
        CDF = d['CDF']
        best_av = d['BEST_AV']
        sig_av = d['SIG_AV']

#        sig_hi_ebv = np.array([d['EBV_1SIG'][1] - d['BEST_EBV'] for d in SN12CU_CHISQ_DATA])
#        sig_ebv = np.array([d['SIG_EBV'] for d in SN12CU_CHISQ_DATA])



#        rvs = np.array([d['BEST_RV'] for d in SN12CU_CHISQ_DATA])
#        sig_lo_rv = np.array([d['BEST_RV'] - d['RV_1SIG'][0] for d in SN12CU_CHISQ_DATA])
#        sig_hi_rv = np.array([d['RV_1SIG'][1] - d['BEST_RV'] for d in SN12CU_CHISQ_DATA])
#        sig_rv = np.array([d['SIG_RV'] for d in SN12CU_CHISQ_DATA])


        best_av = d['BEST_AV']
        sig_av = d['SIG_AV']


        
        ## plots 1- and 2-sigma regions and shades them with different hues.
        plt.contourf(X, Y, 1-CDF, levels=[1-l for l in contour_levels], cmap=mpl.cm.summer, alpha=0.5)

        ## Outlines 1-sigma contour
        C1 = plt.contour(X, Y, CDF, levels=[contour_levels[1]], linewidths=1, colors=['k'], alpha=0.7)
        
        #plt.contour(X, Y, CHISQ-chisq_min, levels=[1.0, 4.0], colors=['r', 'g'])
        
        ## mark minimum
        plt.scatter(best_ebv, best_rv, marker='s', facecolors='r')
        
        # show results on plot
        if i%6==0:
                plttext1 = "Phase: {}".format(phase)
        else:
                plttext1 = "{}".format(phase)
                
        plttext2 = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                   "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                   "\n$A_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
        plttext2 = plttext2.format(best_ebv, maxebv_1sig-best_ebv, best_ebv-minebv_1sig,
                                   best_rv, maxrv_1sig-best_rv, best_rv-minrv_1sig,
                                   best_av, sig_av, sig_av
                                   )
                                           
        if phase not in [11.5, 16.5, 18.5, 21.5]:
                ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                ax.text(.04, .85, plttext2, size=INPLOT_LEGEND_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                ax.axhspan(2.9, (y.max()), facecolor='k', alpha=0.1)
        
        else:
                ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                ax.text(.04, .32, plttext2, size=INPLOT_LEGEND_FONTSIZE,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                ax.axhspan(3.32, (y.max()), facecolor='k', alpha=0.1)
                ax.axhspan((y.min()), 2.5, facecolor='k', alpha=0.1)
                
                
        # format subplot...
        plt.ylim(y.min(), y.max())
        plt.xlim(x.min(), y.max())
        
        ax.set_yticklabels([])
        ax2 = ax.twinx()
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(y.min(), y.max())
        
        if i%6 == 5:
                ax2.set_ylabel('\n$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=5)
        if i%6 == 0:
                ax.set_ylabel('$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=-2)
        if i>=6:
                ax.set_xlabel('\n$E(B-V)$', fontsize=AXIS_LABEL_FONTSIZE)
        
        # format x labels
        labels = ax.get_xticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_xticklabels(labels)
        ax.get_xaxis().set_tick_params(direction='in', pad=-20)
        
        # format y labels
        labels = ax2.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax2.set_yticklabels(labels)
        ax2.get_yaxis().set_tick_params(direction='in', pad=-30)
        
        plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
        plt.setp(ax2.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)


    SNcontour = SNname + '_contour'
    
    if unfilt:
        filenm = SNcontour + '_unfilt.png'
    else: 
        filenm = SNcontour + '_filtered.png'

    plt.savefig(filenm)

    return

def plot_summary(SNname, SN12CU_CHISQ_DATA, unfilt):

    ''' 
     
    
    '''

    
    print "Plotting time dependence", SNname


    phase_index = np.array([d['phase'] for d in SN12CU_CHISQ_DATA])


#np.array([phase_index for phase_index in SN12CU_CHISQ_DATA['phase']])
    print phase_index
   
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
    sig_EBV_overall = np.sqrt(np.sum(sig_ebv**2))/N
    sig_RV_overall = np.sqrt(np.sum(sig_rv**2))/N
    sig_AV_overall = np.sqrt(np.sum(sig_av**2))/N
    sig_del_mu_overall = np.sqrt(np.sum(sig_del_mu**2))/N

    print 'EBV_ovarall_avg, sig_EBV_overall', EBV_overall_avg, sig_EBV_overall   
    print 'RV_ovarall_avg, sig_RV_overall', RV_overall_avg, sig_RV_overall   
    print 'AV_ovarall_avg, sig_AV_overall', AV_overall_avg, sig_AV_overall   
    print 'del_mu_ovarall_avg, sig_del_mu_overall', del_mu_overall_avg, sig_del_mu_overall   


####***** Plotting Time Dependence ***************


    fig = plt.figure(figsize = (15, 12))
    ax = fig.add_subplot(411) 
    plt.errorbar(phase_index, ebvs, yerr = [sig_lo_ebv, sig_hi_ebv], fmt = 'b.')
    plt.plot(phase_index, EBV_overall_avg*np.ones(phase_index.shape), '--')
    ax.fill_between(phase_index, (EBV_overall_avg+sig_EBV_overall)*np.ones(phase_index.shape), (EBV_overall_avg-sig_EBV_overall)*np.ones(phase_index.shape), \
                       facecolor='black', alpha=0.1)
    ebv_txt = "$E(B-V) = {:.3f}\pm{:.3f}$".format(EBV_overall_avg, sig_EBV_overall)
    ax.text(.04, .9, ebv_txt, size=AXIS_LABEL_FONTSIZE,\
                            horizontalalignment='left', \
                            verticalalignment='top', \
                            transform=ax.transAxes)
    plt.ylabel('$E(B-V)$')


    ax = fig.add_subplot(412) 
    plt.errorbar(phase_index, rvs, yerr = [sig_lo_rv, sig_hi_rv], fmt = 'r.')
    plt.plot(phase_index, RV_overall_avg*np.ones(phase_index.shape), '--')
    ax.fill_between(phase_index, (RV_overall_avg+sig_RV_overall)*np.ones(phase_index.shape), (RV_overall_avg-sig_RV_overall)*np.ones(phase_index.shape), \
                       facecolor='black', alpha=0.1)
    rv_txt = "$R_V = {:.3f}\pm{:.3f}$".format(RV_overall_avg, sig_RV_overall)
    ax.text(.04, .9, rv_txt, size=AXIS_LABEL_FONTSIZE,\
                            horizontalalignment='left', \
                            verticalalignment='top', \
                            transform=ax.transAxes)
    plt.ylabel('$R_V$')


    ax = fig.add_subplot(413) 
    plt.errorbar(phase_index, avs, yerr = sig_av, fmt = 'g.')
    plt.plot(phase_index, AV_overall_avg*np.ones(phase_index.shape), '--')
    ax.fill_between(phase_index, (AV_overall_avg+sig_AV_overall)*np.ones(phase_index.shape), (AV_overall_avg-sig_AV_overall)*np.ones(phase_index.shape), \
                       facecolor='black', alpha=0.1)
    av_txt = "$A_V = {:.3f}\pm{:.3f}$".format(AV_overall_avg, sig_AV_overall)
    ax.text(.04, .9, av_txt, size=AXIS_LABEL_FONTSIZE,\
                            horizontalalignment='left', \
                            verticalalignment='top', \
                            transform=ax.transAxes)
    plt.ylabel('$A_V$')

    
    ax = fig.add_subplot(414) 
    plt.errorbar(phase_index, del_mus, yerr = sig_del_mu, fmt = 'b.')
    plt.plot(phase_index, del_mu_overall_avg*np.ones(phase_index.shape), '--')
    ax.fill_between(phase_index, (del_mu_overall_avg+sig_del_mu_overall)*np.ones(phase_index.shape), (del_mu_overall_avg-sig_del_mu_overall)*np.ones(phase_index.shape), \
                       facecolor='black', alpha=0.1)
    del_mu_txt = "$\Delta\mu = {:.3f}\pm{:.3f}$".format(del_mu_overall_avg, sig_del_mu_overall)
    ax.text(.04, .9, del_mu_txt, size=AXIS_LABEL_FONTSIZE,\
                            horizontalalignment='left', \
                            verticalalignment='top', \
                            transform=ax.transAxes)
    plt.ylabel('$\Delta\mu$')



    plt.xlabel('Phases')
    
    SNsummary = SNname + '_summary'
    
    if unfilt:
        filenm = SNsummary + '_unfilt.png'
    else: 
        filenm = SNsummary + '_filtered.png'

    plt.savefig(filenm)



                            

#    for i, phase_index, phase, d in zip(range(len(SN12CU_CHISQ_DATA)), select_phases, [phases[i] for i in select_phases], SN12CU_CHISQ_DATA):
       



if __name__ == "__main__":
    
    
    SNname = 'SN12CU' 
    SNdata = SNname + '_CHISQ_DATA'
    
    unfilt = 1
    
    if unfilt:
        filenm = SNdata + '_unfilt.p'
    else:
        filenm = SNdata + '_filtered.p'
    
    
    SN12CU_CHISQ_DATA = pickle.load(open(filenm, 'rb'))   

    plot_contours(SNname, SN12CU_CHISQ_DATA, unfilt)
    plot_summary(SNname, SN12CU_CHISQ_DATA, unfilt)