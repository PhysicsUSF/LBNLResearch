'''
::Author::
Andrew Stocker

::Description::
This program will plot the color excess plot for 14J and 12CU with various reddening laws.

::Last Modified::
08/01/2014

'''
import loader as l
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sncosmo as snc

from copy import deepcopy
from itertools import izip
from loader import redden_fm, redden_pl, redden_pl2
from pprint import pprint
from scipy.interpolate import interp1d
from sys import argv



################################################################################
### VARS #######################################################################

## vars for phase plot ##
PLOTS_PER_ROW = 6
N_BUCKETS = 20


################################################################################
### 12CU LOADER ################################################################

def load_12cu_colors(phases, filters, zp):
    prefix = zp['prefix']

    # correct for Milky Way extinction
    sn12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)
    sn12cu = l.interpolate_spectra(phases, sn12cu)

    if type(sn12cu) == type(()):  # convert from tuple to list if just one phase
        sn12cu = [sn12cu]
    
    sn12cu_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')/zp['V']) for t in sn12cu]

    sn12cu_colors = {i:{} for i in xrange(len(phases))}
    for f in filters:
        band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)/zp[f]) for t in sn12cu]
        band_colors = np.array(sn12cu_vmags)-np.array(band_mags)
        for i, color in enumerate(band_colors):
            sn12cu_colors[i][f] = color
        
    return sn12cu_colors


################################################################################
### GENERAL PLOTTING FUNCTIONS #################################################


def plot_snake(ax, rng, init, red_law, x, y, CDF, plot2sig=False):
    snake_hi_1sig = deepcopy(init)
    snake_lo_1sig = deepcopy(init)
    if plot2sig:
        snake_hi_2sig = deepcopy(init)
        snake_lo_2sig = deepcopy(init)
    
    for i, EBV in enumerate(x):
        for j, RV in enumerate(y):
            if CDF[j,i]<0.683:
                red_curve = red_law(rng, np.zeros(rng.shape), -EBV, RV, return_excess=True)
                snake_hi_1sig = np.maximum(snake_hi_1sig, red_curve)
                snake_lo_1sig = np.minimum(snake_lo_1sig, red_curve)
            elif plot2sig and CDF[j,i]<0.955:
                red_curve = red_law(rng, np.zeros(rng.shape), -EBV, RV, return_excess=True)
                snake_hi_2sig = np.maximum(snake_hi_2sig, red_curve)
                snake_lo_2sig = np.minimum(snake_lo_2sig, red_curve)
                
    ax.fill_between(10000./rng, snake_lo_1sig, snake_hi_1sig, facecolor='black', alpha=0.3)
    if plot2sig:
        ax.fill_between(10000./rng, snake_lo_2sig, snake_hi_2sig, facecolor='black', alpha=0.1)


################################################################################

def plot_phase_excesses(name, loader, filters, zp):
    
    try:
        print "Loading sn2012cu data from 'sn12cu_chisq_data.pkl' ..."
        SN12CU_CHISQ_DATA = pickle.load(open('sn12cu_chisq_data.pkl', 'rb'))
    except:
        print "Failed.  Fetching data ..."
        from plot_excess_contours import get_12cu_best_ebv_rv
        SN12CU_CHISQ_DATA = get_12cu_best_ebv_rv()
    
    
    print "Plotting excesses of",name," with best fit from contour..."
    
    prefix = zp['prefix']
    phases = [d['phase'] for d in SN12CU_CHISQ_DATA]
    ref = loader(phases, filters, zp)
    filter_eff_waves = [snc.get_bandpass(prefix+f).wave_eff for f in filters]

    # get 11fe synthetic photometry at BMAX, get ref sn color excesses at BMAX
    sn11fe = l.interpolate_spectra(phases, l.get_11fe())
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    pmin, pmax = np.min(phases), np.max(phases)
    
    for i, d, sn11fe_phase in izip(xrange(len(SN12CU_CHISQ_DATA)), SN12CU_CHISQ_DATA, sn11fe):
        phase = d['phase']
        print "Plotting phase {} ...".format(phase)
        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        
        # calculate sn11fe band magnitudes
        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)/zp[f])
                       for f in filters}
        
        # calculate V-X colors for sn11fe
        sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]

        # make list of colors of reference supernova for given phase index i
        ref_colors = [ref[i][f] for f in filters]

        # get colors excess of reference supernova compared for sn11fe
        phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)

        # convert effective wavelengths to inverse microns then plot
        eff_waves_inv = (10000./np.array(filter_eff_waves))
        mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))    
        plt.plot(eff_waves_inv, phase_excesses, 's', color=mfc_color,
                 ms=8, mec='none', mfc=mfc_color, alpha=0.8)
        
        # reddening law vars
        red_law = redden_fm
        linestyle = '--'

        x = np.arange(3000,10000,10)
        xinv = 10000./x
        red_curve = red_law(x, np.zeros(x.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)
        redln, = plt.plot(xinv, red_curve, 'k'+linestyle)
        
        test_red_curve = red_law(np.array(filter_eff_waves), np.zeros(np.array(filter_eff_waves).shape),
                                 -d['BEST_EBV'], d['BEST_RV'], return_excess=True)
        
        plot_snake(ax, x, red_curve, red_law, d['x'], d['y'], d['CDF'])
        
        plttext = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$A_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
                          
        plttext = plttext.format(d['BEST_EBV'], d['EBV_1SIG'][1]-d['BEST_EBV'], d['BEST_EBV']-d['EBV_1SIG'][0],
                                 d['BEST_RV'], d['RV_1SIG'][1]-d['BEST_RV'], d['BEST_RV']-d['RV_1SIG'][0],
                                 d['BEST_AV'], d['AV_1SIG'][1]-d['BEST_AV'], d['BEST_AV']-d['AV_1SIG'][0]
                                 )
        
        ax.text(.95, .95, plttext, size=16,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        
        ax.set_title('phase: '+str(phase))
        
        if i%PLOTS_PER_ROW == 0:
            plt.ylabel('$E(V-X)$')
        if i>=(numrows-1)*PLOTS_PER_ROW:
            plt.xlabel('Wavelength ($1 / \mu m$)')
        plt.xlim(1.0, 3.0)
        plt.ylim(-3.0, 2.0)


################################################################################
### MAIN #######################################################################


def main():
    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
    
    fig = plt.figure()
    
    plot_phase_excesses('SN2012CU', load_12cu_colors, filters_bucket, zp_bucket)
    
    fig.suptitle('SN2012CU: Color Excess Per Phase', fontsize=18)
                 
    p1, = plt.plot(np.array([]), np.array([]), 'k--')
    fig.legend([p1], ['Fitzpatrick-Massa (1999)'], loc=1, bbox_to_anchor=(0, 0, .905, .975))
    
    plt.show()
    
if __name__=='__main__':
    main()
