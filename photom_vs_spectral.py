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
STEPS = 100
#N_BUCKETS = 20

EBV_GUESS = 1.1
EBV_PAD = .3
RV_GUESS = 2.75
RV_PAD = .75


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15






if __name__ == "__main__":

    '''
    
    python photom_vs_spectral.py -N_BUCKETS 20
    
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-N_BUCKETS', type = int)

    args = parser.parse_args()
    print 'args', args
    N_BUCKETS = args.N_BUCKETS
    hi_wave = 9700.
    lo_wave = 3300.
    

    ## Setting up tophat filters
    filters_bucket, zp_bucket, LOW_wave, HIGH_wave = l.generate_buckets(lo_wave, hi_wave, N_BUCKETS)  #, inverse_microns=True)
    filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff for f in filters_bucket])

    del_wave = (HIGH_wave  - LOW_wave)/N_BUCKETS

    sn12cu_excess, phases, sn11fe, sn12cu, prefix = load_12cu_excess(filters_bucket, zp_bucket, del_wave, AB_nu = True)
    #exit(1)


    ref_wave = sn11fe[0][1].wave

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
    plt.show()
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

    plt.show()



    fig = plt.figure(figsize = (10, 8))

    for i, phase in enumerate(['-6.5']):  # enumerate(phases)
        print "Plotting phase {} ...".format(phase)
        
        ax = plt.subplot(111)  # ax = plt.subplot(2,6,i+1)
        #                print 'sn12cu_excess[i].shape', sn12cu_excess[i].shape
        #                exit(1)
        plot_contour(i, phase, redden_fm, sn12cu_excess[i], filter_eff_waves,
                     EBV_GUESS, EBV_PAD, RV_GUESS, RV_PAD, STEPS, ax)


    fig.subplots_adjust(left=0.04, bottom=0.08, right=0.95, top=0.92, hspace=.06, wspace=.1)
    fig.suptitle('SN2012CU: $E(B-V)$ vs. $R_V$ Contour Plot per Phase', fontsize=TITLE_FONTSIZE)
    plt.show()




#    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
#    
#    #        print 'filters_bucket', filters_bucket
#    #        print 'zp_bucket', zp_bucket
#    #        exit(1)
#    
#    filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff for f in filters_bucket])
#
#    for f in filters_bucket:
#        band_wave = snc.get_bandpass(zp_bucket['prefix']+f).wave
#        band_wave_grad = np.gradient(band_wave)
#        band_trans = np.array([snc.get_bandpass(zp_bucket['prefix']+f).trans for f in filters_bucket])
##        weights = band_trans * band_wave_grad
##            return np.sum(self.wave * weights) / np.sum(weights)
#
#    phases = load_12cu_excess_X(filters_bucket, zp_bucket)

    

