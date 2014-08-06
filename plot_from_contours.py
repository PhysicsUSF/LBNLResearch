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
import sncosmo as snc

from itertools import izip
from loader import redden_fm, redden_pl, redden_pl2
from pprint import pprint
from scipy.interpolate import interp1d
from sys import argv



################################################################################
### VARS #######################################################################

## vars for phase plot ##
PLOTS_PER_ROW = 6
FONT_SIZE = 14
N_BUCKETS = 20


################################################################################
### LOADERS ####################################################################

def load_14j_colors(phases, filters, zp):
    sn14j  = l.get_14j()
    
    # get 14j photometry at BMAX
    sn14j_colors = {i:{} for i in xrange(len(phases))}
    for f in filters:
        band_phases = np.array([d['phase'] for d in sn14j[f]])
        try:
            band_colors = np.array([(d['Vmag']-d['AV'])-(d['mag']-d['AX']) for d in sn14j[f]])
        except:
            band_colors = np.array([0.0 for d in sn14j[f]])

        sn14j_int = interp1d(band_phases, band_colors)
        for i, phase in enumerate(phases):
            sn14j_colors[i][f] = float(sn14j_int(phase))
        
    return sn14j_colors


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
### GENERAL PLOTTING FUNCTION ##################################################


def plot_phase_excesses(name, loader, filters, zp):
                            
    from plot_excess_contours import get_12cu_best_ebv_rv
    phases, EBVS, RVS = get_12cu_best_ebv_rv()
    
    print phases
    print EBVS
    print RVS
    
    print "Plotting excesses of",name," with best fit from contour..."
    
    ref = loader(phases, filters, zp)
    prefix = zp['prefix']
    filter_eff_waves = [snc.get_bandpass(prefix+f).wave_eff for f in filters]

    # get 11fe synthetic photometry at BMAX, get ref sn color excesses at BMAX
    sn11fe = l.interpolate_spectra(phases, l.get_11fe())

    if type(sn11fe) == type(()):  # convert from tuple to list if just one phase
        sn11fe = [sn11fe]
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    
    pmin, pmax = np.min(phases), np.max(phases)
    
    for i, phase, sn11fe_phase in izip(xrange(len(phases)), phases, sn11fe):
        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        
        # calculate sn11fe band magnitudes
        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)/zp[f])
                       for f in filters}
        
        # calculate V-X colors for sn11fe
        sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]

        # make list of colors of reference supernova for given phase i
        ref_colors = [ref[i][f] for f in filters]

        # get colors excess of reference supernova compared for sn11fe
        phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)

        # convert effective wavelengths to inverse microns then plot
        eff_waves_inv = (10000./np.array(filter_eff_waves))
        mfc_color = plt.cm.gist_rainbow((phase-pmin)/(pmax-pmin))    
        plt.plot(eff_waves_inv, phase_excesses, 's', color=mfc_color,
                 ms=8, mec='none', mfc=mfc_color, alpha=0.8)
        
            
        red_law = redden_fm
        linestyle = '--'
        llabel = 'Fitzpatrick-Massa'
        EBV = -EBVS[i]

                
        x = np.arange(3000,10000,10)
        xinv = 10000./x
        red_curve = red_law(x, np.zeros(x.shape), EBV, RVS[i], return_excess=True)
        redln, = plt.plot(xinv, red_curve, 'k'+linestyle, label=llabel)
        
        plt_text = '$E(B-V)$: $'+str(round(EBVS[i],2))+'$' + \
                   '\n'+'$R_V$: $'+str(round(RVS[i],2))+'$'
                
        ax.text(.95, .95, plt_text, size=FONT_SIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        
        ax.set_title('phase: '+str(phase))
        ax.legend(loc=3, prop={'size':FONT_SIZE})
                     
        if i%PLOTS_PER_ROW == 0:
            plt.ylabel('$E(V-X)$')
        if i>=(numrows-1)*PLOTS_PER_ROW:
            plt.xlabel('Wavelength ($1 / \mu m$)')
        plt.xlim(1.0, 3.0)


################################################################################
### MAIN #######################################################################


def main(filters, zp):
    
    fig = plt.figure()
    plot_phase_excesses('SN2012CU', load_12cu_colors, filters, zp)
        
    fig.suptitle('SN2012CU: Color Excess Per Phase (with best fit for $R_V$)',
                 fontsize=18)
    
    
    
if __name__=='__main__':
    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)

    main(filters_bucket, zp_bucket)
    
    plt.show()
    
