'''
::Author::
Xiaosheng Huang
(based on Andrew's plot_excess_contour.py)


Date: 12/3/14
Purpose: calculate effective AB mag for narrow bands equally spaced in inverse wavelength and compare that with the single-wavelength AB_nu (see below).


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

from plot_excess_contours import *
from mag_spectrum_fitting import ABmag_nu



### CONST ###
c = 3e18  # speed of light in A/sec.

### VARS ###



EBV_GUESS = 1.1
EBV_PAD = .3
RV_GUESS = 2.75
RV_PAD = .75


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15

PLOTS_PER_ROW = 6

V_wave = 5413.5

'''This function needs to be adapted for my way of doing things:
    
    1. pass sn11fe_colors (instead of recalculating it)
    2. pass sn12cu_colors (note: there is no need to de-redden)
    3. Do we need zp??

'''





#def plot_phase_excesses(name, loader, SN12CU_CHISQ_DATA, red_law, filters, zp):
def plot_phase_excesses(name, sn11fe_colors, sn12cu_colors, filter_eff_waves, SN12CU_CHISQ_DATA, filters, red_law, phases, pmin, pmax, rv_spect, ebv_spect):
    
    
    print "Plotting excesses of",name," with best fit from contour..."
    
    #    prefix = zp['prefix']
    #   phases = [d['phase'] for d in SN12CU_CHISQ_DATA]
    #   ref = loader(phases, filters, zp)
    #filter_eff_waves = [snc.get_bandpass(prefix+f).wave_eff for f in filters]

    # get 11fe synthetic photometry at BMAX, get ref sn color excesses at BMAX
    #sn11fe = l.interpolate_spectra(phases, l.get_11fe())
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    #pmin, pmax = np.min(phases), np.max(phases)
    
    #    for i, d, sn11fe_phase in izip(xrange(len(SN12CU_CHISQ_DATA)), SN12CU_CHISQ_DATA, sn11fe):
    for i, d, phase in izip(xrange(len(SN12CU_CHISQ_DATA)), SN12CU_CHISQ_DATA, phases):
        #phase = d['phase']
        
        print "Plotting phase {} ...".format(phase)
        #exit(1)
            
            
            #ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        ax = plt.subplot(111)
            
        # calculate sn11fe band magnitudes.  Note since the method bandflux() returns the mag and the error, I have added [0] to select the total flux.  -XH
        #        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)[0]/zp[f])
        #               for f in filters}
#        print 'len(sn11fe_mags)', len(sn11fe_mags)
#        print 'sn11fe_mags', sn11fe_mags




        # calculate V-X colors for sn11fe
        #        sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]
        
        # make list of colors of reference supernova for given phase index i.  Again each ref[i][f] is an array of two number: mag and flux error converted to mag (via -2.5*log(flux error), which is not the right way); I use [0] to select the mag for now.  -XH
#        print 'sn11fe_colors', sn11fe_colors
#        print 'sn11fe_colors[0]', sn11fe_colors[0]


        
        ref_colors = [sn11fe_colors[i][f] for f in filters]
        SN_colors = [sn12cu_colors[i][f] for f in filters]

#        print 'ref_colors', ref_colors
#        exit(1)

        # get colors excess of reference supernova compared for sn11fe
        phase_excesses = np.array(SN_colors)-np.array(ref_colors)
  
        
        # convert effective wavelengths to inverse microns then plot
        #eff_waves_inv = 10000./filter_eff_waves
        mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))    
            #plt.plot(filter_eff_waves, phase_excesses, 's', color=mfc_color, ms=8, mec='none', mfc=mfc_color, alpha=0.8)

        plt.plot(filter_eff_waves, phase_excesses, 's', color='black', ms=8) #, mec='none', mfc=mfc_color, alpha=0.8)

#        print 'eff_waves_inv', eff_waves_inv
#        print 'phase_excesses', phase_excesses
#        plt.show()
#        exit(1)

        # reddening law vars
        linestyle = '--'
        
        x = np.arange(3000,10000,10)
        #xinv = 10000./x
        red_curve = red_law(x, np.zeros(x.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)
        plt.plot(x, red_curve, 'k'+linestyle)
        slo, shi = plot_snake(ax, x, red_curve, red_law, d['x'], d['y'], d['CDF'])

#        ebv_spect = 1.01
#        rv_spect = 2.85
        red_curve_spect = red_law(x, np.zeros(x.shape), -ebv_spect, rv_spect, return_excess=True)
        plt.plot(x, red_curve_spect, 'r-')

        ## plot where V band is.   -XH
        plt.plot([x.min(), x.max()], [0, 0] ,'--')
        plt.plot([V_wave, V_wave], [red_curve.min(), red_curve.max()] ,'--')



        # TEST #####################
#        dd = SN12CU_CHISQ_DATA_PL[i]
#        test_red_curve = redden_pl2(x, np.zeros(x.shape), -dd['BEST_EBV'], dd['BEST_RV'], return_excess=True)
#        plt.plot(xinv, test_red_curve, 'r-')
#        AV = dd['BEST_EBV']*dd['BEST_RV']
#        P = np.log((1/dd['BEST_RV'])+1)/np.log(0.8)
#        W.writerow([phase, AV, P])
        ############################


        #pprint( zip([int(f) for f in filter_eff_waves],
                #[round(f,2) for f in 10000./np.array(filter_eff_waves)],
                #filters,
                #[round(p,2) for p in phase_excesses],
                #[round(r,2) for r in shi(filter_eff_waves)],
                #[round(r,2) for r in slo(filter_eff_waves)],
                #[round(r,2) for r in interp1d(x, test_red_curve)(filter_eff_waves)]
                #)
               #)
        
        
        # print data on subplot
        plttext = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$A_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
        
        plttext = plttext.format(d['BEST_EBV'], d['EBV_1SIG'][1]-d['BEST_EBV'], d['BEST_EBV']-d['EBV_1SIG'][0],
                                 d['BEST_RV'], d['RV_1SIG'][1]-d['BEST_RV'], d['BEST_RV']-d['RV_1SIG'][0],
                                 d['BEST_AV'], d['AV_1SIG'][1]-d['BEST_AV'], d['BEST_AV']-d['AV_1SIG'][0]
                                 )
        
        ax.text(.95, .2, plttext, size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        
        # format subplot
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




    # TEST #####################
#F.close()
    ############################


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



if __name__ == "__main__":

    '''
    
    python photom_vs_spectral.py -select_phases 1 -N_BUCKETS 20 -RV_STEPS 41 -EBV_STEPS 41 -ebv_spect 1.00 -rv_spect 2.8
    
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-N_BUCKETS', type = int)
    parser.add_argument('-RV_STEPS', type = int)
    parser.add_argument('-EBV_STEPS', type = int)
    parser.add_argument('-select_phases',  '--select_phases', nargs='+', type=int)  # this can take a tuple: -select_phases 0 4  but the rest of the program can't handle more than
                                                                                    # one phases yet.  -XH 12/7/14
    _ = parser.add_argument('-ebv_spect', type = float)  # just another way to add an argument to the list.
    _ = parser.add_argument('-rv_spect', type = float)  # just another way to add an argument to the list.


    args = parser.parse_args()
    print 'args', args
    N_BUCKETS = args.N_BUCKETS
    RV_STEPS = args.RV_STEPS
    EBV_STEPS = args.EBV_STEPS
    select_phases = np.array(args.select_phases) ## if there is only one phase select, it needs to be in the form of a 1-element array for all things to work.
    ebv_spect = args.ebv_spect
    rv_spect = args.rv_spect

    hi_wave = 9700.
    lo_wave = 3300.
    

    ## Setting up tophat filters
    filters_bucket, zp_bucket, LOW_wave, HIGH_wave = l.generate_buckets(lo_wave, hi_wave, N_BUCKETS)  #, inverse_microns=True)
    filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff for f in filters_bucket])

    del_wave = (HIGH_wave  - LOW_wave)/N_BUCKETS


#select_phases = np.array([1])
    sn12cu_excess, phases, sn11fe, sn12cu,  sn12cu_colors, sn11fe_colors, prefix = load_12cu_excess(select_phases, filters_bucket, zp_bucket, del_wave, AB_nu = True)


    ref_wave = sn11fe[0][1].wave


    
#    for phase_index in select_phases:
#        print 'phase_index', phase_index
#
#    print phases
#    exit(1)
#    print phases[select_phases]
#    print [phases[select_phases]]
#
#    exit(1)


#    for phase in [phases[select_phases]]:
#        print 'phase', phase
#
#    exit(1)
    #for phase, sn11fe_phase in izip(select_phases, sn11fe[select_phases]):

#eventually want to use the same for loop for 11fe and 12cu -- turn it into a function that returns mag's.  Use SN = sn11fe or sn12cu in the function call.
#    for phase_index, phase in zip(select_phases, [phases[select_phases]]):
#        
#        print '\n\n\n Phase_index', phase_index, '\n\n\n'
#        sn11fe_phase = sn11fe[phase_index]

    for i, phase, sn11fe_phase in izip(xrange(1), phases, sn11fe):
        print 'phase', phase
        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f, del_wave = del_wave, AB_nu = True)[0]) - 48.6 for f in filters_bucket}
        sn11fe_only_mags = np.array([sn11fe_mags[f] for f in filters_bucket])
        #sn11fe_1phase = sn11fe[i]
        flux_11fe = sn11fe_phase[1].flux
        mag_11fe = ABmag_nu(flux_11fe, ref_wave)

# convert effective wavelengths to inverse microns then plot
#eff_waves_inv = (10000./np.array(filter_eff_waves))
    pmin, pmax = np.min(phases), np.max(phases)
    mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))
    plt.figure()
    plt.plot(filter_eff_waves, sn11fe_only_mags, 's', ms=8, mec='none')
    plt.plot(ref_wave, mag_11fe, 'r.')

    
#    plt.figure()
#    plt.plot(np.array(filter_eff_waves), sn12cu_only_mags, 's', ms=8, mec='none')
#    plt.plot(ref_wave, mag_12cu, 'k.')
#    

#exit(1)

    for i, phase, sn12cu_phase in izip(xrange(1), phases, sn12cu):
        print 'phase', phase
        sn12cu_mags = {f : -2.5*np.log10(sn12cu_phase[1].bandflux(prefix+f, del_wave = del_wave, AB_nu = True)[0]) - 48.6 for f in filters_bucket}
        sn12cu_only_mags = np.array([sn12cu_mags[f] for f in filters_bucket])
        #sn12cu_1phase = sn12cu[i]
        #flux_12cu = sn12cu_phase[1].flux
        flux12cu_interp = interp1d(sn12cu_phase[1].wave, sn12cu_phase[1].flux)
        mag_12cu = ABmag_nu(flux12cu_interp(ref_wave), ref_wave)


    plt.figure()
    plt.plot(filter_eff_waves, sn12cu_only_mags, 's', ms=8, mec='none')
    plt.plot(ref_wave, mag_12cu, 'k.')




    fig = plt.figure(figsize = (10, 8))
    SN12CU_CHISQ_DATA = []
    for i, phase in enumerate(select_phases):  # enumerate(phases)
        print "Plotting phase {} ...".format(phase)
        
        ax = plt.subplot(111)  # ax = plt.subplot(2,6,i+1)
        #                print 'sn12cu_excess[i].shape', sn12cu_excess[i].shape
        #                exit(1)
        
        
        ####**** NOTE: plot_contour() is called twic here.   Fix this *****####
        
        
        ## plot_contour() is where chi2 is calculated.
        plot_contour(i, phase, redden_fm, sn12cu_excess[i], filter_eff_waves,
                     EBV_GUESS, EBV_PAD, RV_GUESS, RV_PAD, RV_STEPS, EBV_STEPS, ax)


        x, y, CDF, \
        best_ebv, best_rv, best_av, \
            ebv_1sig, ebv_2sig, \
            rv_1sig, rv_2sig, \
            av_1sig, av_2sig = plot_contour(i, phase, redden_fm, sn12cu_excess[i],
                                            filter_eff_waves, EBV_GUESS,
                                            EBV_PAD, RV_GUESS, RV_PAD, RV_STEPS, EBV_STEPS
                                            )
        
        SN12CU_CHISQ_DATA.append({'phase'   : phase,
                                 'x'       : x,
                                 'y'       : y,
                                 'CDF'     : CDF,
                                 'BEST_EBV': best_ebv,
                                 'BEST_RV' : best_rv,
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
    plot_phase_excesses('SN2012CU', sn11fe_colors, sn12cu_colors, filter_eff_waves, SN12CU_CHISQ_DATA, filters_bucket, redden_fm, phases, pmin, pmax, rv_spect, ebv_spect)

    plt.show()






