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
    


- I have changed 'vega' to 'ab' in loader.py.  1) to make easier comparison between photometric fit and spectral fit.  2) I ran into an error with 'vega' -- it seems that it can't retrieve data from a ftp site.  It really ought to be changed here and on the command line.

- Even with AB mag, I have recovered low RV values.  The last phase still looks different from other phases.

- At 40 bins or above, one need to remove nan to make it work.  - Dec 2. 2014



::Description::
This program will fit RV based on color excess of 2012cu

::Last Modified::
08/01/2014

'''
import loader as l
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from itertools import izip
from loader import redden_fm, redden_pl, redden_pl2
from pprint import pprint
from scipy.stats import chisquare


### CONST ###
c = 3e18  # speed of light in A/sec.

### VARS ###
STEPS = 100
N_BUCKETS = 20

EBV_GUESS = 1.1
EBV_PAD = .3
RV_GUESS = 2.75
RV_PAD = .75


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15


################################################################################


def ABmag_nu(flux_per_Hz):
    return -2.5*np.log10(flux_per_Hz) - 48.6  ## Bessell & Murphy 2012.



def load_12cu_excess(filters, zp):


    prefix = zp['prefix']  # This specifies units as inverse micron or angstrom; specified in the function call to l.generate_buckets().
    
    print 'filters', filters
    print "zp['V']", zp['V']  # the bands are numbered except V band, which is labeled as 'V'.
    
    
    EXCESS = {}


    ## load spectra, interpolate 11fe to 12cu phases (only first 12)

    # correct for Milky Way extinction
    sn12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)
    sn12cu = filter(lambda t: t[0]<28, sn12cu)   # here filter() is a python built-in function.
    phases = [t[0] for t in sn12cu]
    
    #      print 'sn12cu[1]',
    
    ## the method bandflux() returns (bandflux, bandfluxerr), see spectral.py
    ## Further need to understandhow bandflux and bandfluerr are calcuated in spectral.py.
    ## It seems that there may be a conversion between photon flux and energy flux.
    sn12cu_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')[0]/zp['V']) for t in sn12cu]

    sn12cu_colors = {i:{} for i in xrange(len(phases))}
    for f in filters:
            band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)[0]/zp[f]) for t in sn12cu]
            band_colors = np.array(sn12cu_vmags)-np.array(band_mags)
            for i, color in enumerate(band_colors):
                    sn12cu_colors[i][f] = color
    

#sn11fe = l.interpolate_spectra(phases, l.get_11fe())

    sn11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))

    ref_wave = sn11fe[0][1].wave


    for i, phase, sn11fe_phase in izip(xrange(1), phases, sn11fe):
            print 'phase', phase
            sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)[0]/zp[f])
                           for f in filters}
            sn11fe_only_mags = np.array([sn11fe_mags[f] for f in filters])
            sn11fe_1phase = sn11fe[i]
            flux = sn11fe_1phase[1].flux
            mag = ABmag_nu(flux*ref_wave**2/c)
#    plt.figure()
#plt.plot(ref_wave, flux, 'k.')
#    plt.figure()
#    plt.plot(ref_wave, mag, 'r.')
#    plt.show()
#    exit(1)
    print 'sn11fe_only_mags', sn11fe_only_mags

    
    # convert effective wavelengths to inverse microns then plot
    eff_waves_inv = (10000./np.array(filter_eff_waves))
    pmin, pmax = np.min(phases), np.max(phases)
    mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))
    plt.plot(eff_waves_inv, sn11fe_only_mags, 's', ms=8, mec='none')
    plt.plot(1e4/ref_wave, mag, 'r.')
    plt.show()


    print 'sn11fe_mags', sn11fe_mags
    return phases



if __name__ == "__main__":

    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
    
    #        print 'filters_bucket', filters_bucket
    #        print 'zp_bucket', zp_bucket
    #        exit(1)
    
    
    filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff
                                 for f in filters_bucket])
        
    phases = load_12cu_excess(filters_bucket, zp_bucket)
    
    

