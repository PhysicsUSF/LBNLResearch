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

        ## Outlines 1-sigma contour
        C1 = plt.contour(X, Y, CDF, levels=[contour_levels[1]], linewidths=1, colors=['k'], alpha=0.7)
        
        #plt.contour(X, Y, CHISQ-chisq_min, levels=[1.0, 4.0], colors=['r', 'g'])
        
        ## mark minimum
        plt.scatter(best_ebv, best_rv, marker='s', facecolors='r')
        
        # show results on plot
        if phase_index%6==0:
                plttext1 = "Phase: {}".format(phase)
        else:
                plttext1 = "{}".format(phase)
                
        plttext2 = "$E(B-V)={:.2f}^{{{:+.2f}}}_{{{:+.2f}}}$" + \
                   "\n$R_V={:.2f}^{{{:+.2f}}}_{{{:+.2f}}}$" + \
                   "\n$A_V={:.2f}^{{{:+.2f}}}_{{{:+.2f}}}$"
        
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
        ax.locator_params(nbins=4)  # this sets the number of tickmarks at 5.
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

    SNcontour = SNname + '_contour'
    
    if unfilt:
        filenm = SNcontour + '_unfilt.png'
    else: 
        filenm = SNcontour + '_filtered.png'

    plt.savefig(filenm)

    return


#def plot_phase_excesses(SN12CU_CHISQ_DATA, redden_fm, snake = snake):

def plot_phase_excesses(SNname, SN12CU_CHISQ_DATA, red_law, unfilt, snake = True):

    ''' 
     
    
    '''
    wave = SN12CU_CHISQ_DATA[0]['WAVE']
    phases = np.array([d['phase'] for d in SN12CU_CHISQ_DATA])

    PLOTS_PER_ROW = math.ceil(len(SN12CU_CHISQ_DATA)/2.)

    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    
    print "Plotting excesses of", SNname, " with best fit from contour..."
    
    fig = plt.figure(figsize = (24, 15))
    
    #numrows = (len(EXCESS)-1)//PLOTS_PER_ROW + 1
    ## Keep; may need this later: pmin, pmax = np.min(phases), np.max(phases)
    
    ## may need this for running all phases    for i, d, sn11fe_phase in izip(xrange(len(SN12CU_CHISQ_DATA)), SN12CU_CHISQ_DATA, sn11fe):
    ##for i, phase, d in zip(range(len(SN12CU_CHISQ_DATA)), phases, SN12CU_CHISQ_DATA):
    for phase_index, phase, d in zip(range(len(phases)), phases, SN12CU_CHISQ_DATA):



        print "Plotting phase {} ...".format(phase)
        print 'phase_index', phase_index            
        

        ax = plt.subplot(numrows, PLOTS_PER_ROW, phase_index + 1)   


        


        phase_excess = d['EXCESS']   #[phase_index][j] for j, f in enumerate(filters)])
        phase_excess_var = d['EXCESS_VAR']  #  np.array([EXCESS_VAR[phase_index][j] for j, f in enumerate(filters)])



        ## Keep the following two line in case I want to plot the symbols with different colors for different phases.  12/10/14
        #mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))
        #plt.plot(filter_eff_waves, phase_excesses, 's', color=mfc_color, ms=8, mec='none', mfc=mfc_color, alpha=0.8)


        plt.errorbar(wave, phase_excess - d['BEST_u'], np.sqrt(phase_excess_var), fmt='r.', ms = 8, label=u'excess', alpha = 0.3) #, 's', color='black', ms=8) #, mec='none', mfc=mfc_color, 
 
 
        ## plot best-fit reddening curve and uncertainty snake
        
        reg_wave = np.arange(3000,10000,10)
        #xinv = 10000./x # can probably delete this soon.  12/18/14
        red_curve = red_law(reg_wave, np.zeros(reg_wave.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)
        plt.plot(reg_wave, red_curve, 'k--')


        if snake:
            ax.fill_between(reg_wave, d['lo_1sig'], d['hi_1sig'], facecolor='black', alpha=0.5)
            ax.fill_between(reg_wave, d['lo_2sig'], d['hi_2sig'], facecolor='black', alpha=0.3)



        ## plot where V band is.   -XH
        plt.plot([reg_wave.min(), reg_wave.max()], [0, 0] ,'--')
        plt.plot([V_wave, V_wave], [red_curve.min(), red_curve.max()] ,'--')
        
    

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
                      "\n$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                      "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                      "\n$A_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                      "\n$u={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
                      
                      

            plttext = plttext.format(d['CHI2_DOF_MIN'],
                                     d['BEST_EBV'], d['EBV_1SIG'][1]-d['BEST_EBV'], d['BEST_EBV']-d['EBV_1SIG'][0],
                                     d['BEST_RV'], d['RV_1SIG'][1]-d['BEST_RV'], d['BEST_RV']-d['RV_1SIG'][0],
                                     d['BEST_AV'], d['SIG_AV'], d['SIG_AV'],
                                     d['BEST_u'], d['SIG_U'][1], d['SIG_U'][0])
        else:
            plttext = "\n$\chi_{{min}}^2/dof = {:.2f}$" + \
                      "\n$E(B-V)={:.2f}$" + \
                      "\n$R_V={:.2f}$" + \
                      "\n$A_V={:.2f}$" + \
                      "\n$u={:.2f}$"
                      
                      

            plttext = plttext.format(d['CHI2_DOF_MIN'],
                                     d['BEST_EBV'],
                                     d['BEST_RV'],
                                     d['BEST_AV'],
                                     d['BEST_u'])






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
            ax.set_title('Phase: {}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel('$E(V-X)$', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_title('{}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
    
        plt.xlim(3000, 10000)
        plt.ylim(-3.0, 2.0)
        
        labels = ax.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_yticklabels(labels)
        
        ax.locator_params(axis = 'x', nbins=4)  # this sets the number of tickmarks at 5.
        labels = ax.get_xticks().tolist()
        labels = [int(label) for label in labels]
        labels[0] = labels[-1] = ''
        major_formatter = FormatStrFormatter('%4.0f')
        ax.xaxis.set_major_formatter(major_formatter)
        ax.set_xticklabels(labels)
        
        plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
        plt.setp(ax.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)

        ## Trimming margins so that more space is used for the plots, and controls space between subplots.
        fig.subplots_adjust(left=0.04, bottom=0.08, right=0.95, top=0.92, hspace=.12, wspace=.16)



    SNsummary = SNname + '_excess'
    
    if unfilt:
        filenm = SNsummary + '_unfilt.png'
    else: 
        filenm = SNsummary + '_filtered.png'

    plt.savefig(filenm)
            


    return # plot_excess()





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
    plt.ylabel('$E(B-V)$', fontsize=AXIS_LABEL_FONTSIZE)


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
    plt.ylabel('$R_V$', fontsize=AXIS_LABEL_FONTSIZE)


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
    plt.ylabel('$A_V$', fontsize=AXIS_LABEL_FONTSIZE)

    
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
    plt.ylabel('$\Delta\mu$', fontsize=AXIS_LABEL_FONTSIZE)



    plt.xlabel('Phases', fontsize=AXIS_LABEL_FONTSIZE)
    
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
    plot_phase_excesses(SNname, SN12CU_CHISQ_DATA, redden_fm, unfilt, snake = True)
