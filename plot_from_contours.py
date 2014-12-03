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
RED_LAW = redden_fm

TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15

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
        
    return interp1d(rng, snake_lo_1sig), interp1d(rng, snake_hi_1sig)


################################################################################

def plot_phase_excesses(name, loader, red_law, filters, zp):
    
    try:
        print "Loading sn2012cu data from 'sn12cu_chisq_data.pkl' ..."
        SN12CU_CHISQ_DATA = pickle.load(open('sn12cu_chisq_data.pkl', 'rb'))
    except:
        print "Failed.  Fetching data ..."
        from plot_excess_contours import get_12cu_best_ebv_rv
        SN12CU_CHISQ_DATA = get_12cu_best_ebv_rv(red_law, filters, zp)
    
    
    # TEST #####################
    import csv
    from plot_excess_contours import get_12cu_best_ebv_rv
    SN12CU_CHISQ_DATA_PL = get_12cu_best_ebv_rv(redden_pl2, filters, zp)
    F = open('12cu_powerlaw_fits.csv', 'w')
    W = csv.writer(F)
    ############################
    
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
        
        # calculate sn11fe band magnitudes.  Note since the method bandflux() returns the mag and the error, I have added [0] to select the total flux.  -XH
        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)[0]/zp[f])
                       for f in filters}
#        print 'len(sn11fe_mags)', len(sn11fe_mags)
#        print 'sn11fe_mags', sn11fe_mags




        # calculate V-X colors for sn11fe
        sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]
        
        # make list of colors of reference supernova for given phase index i.  Again each ref[i][f] is an array of two number: mag and flux error converted to mag (via -2.5*log(flux error), which is not the right way); I use [0] to select the mag for now.  -XH
        ref_colors = [ref[i][f][0] for f in filters]
#        print 'ref_colors', ref_colors
#        exit(1)

        # get colors excess of reference supernova compared for sn11fe
        phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)
  
        
        # convert effective wavelengths to inverse microns then plot
        eff_waves_inv = (10000./np.array(filter_eff_waves))
        mfc_color = plt.cm.cool((phase-pmin)/(pmax-pmin))    
        plt.plot(eff_waves_inv, phase_excesses, 's', color=mfc_color,
                 ms=8, mec='none', mfc=mfc_color, alpha=0.8)
#        print 'eff_waves_inv', eff_waves_inv
#        print 'phase_excesses', phase_excesses
#        plt.show()
#        exit(1)

        # reddening law vars
        linestyle = '--'
        
        x = np.arange(3000,10000,10)
        xinv = 10000./x
        red_curve = red_law(x, np.zeros(x.shape), -d['BEST_EBV'], d['BEST_RV'], return_excess=True)
        plt.plot(xinv, red_curve, 'k'+linestyle)
        slo, shi = plot_snake(ax, x, red_curve, red_law, d['x'], d['y'], d['CDF'])

        
        
        
        # TEST #####################
        dd = SN12CU_CHISQ_DATA_PL[i]
        test_red_curve = redden_pl2(x, np.zeros(x.shape), -dd['BEST_EBV'], dd['BEST_RV'], return_excess=True)
        plt.plot(xinv, test_red_curve, 'r-')
        AV = dd['BEST_EBV']*dd['BEST_RV']
        P = np.log((1/dd['BEST_RV'])+1)/np.log(0.8)
        W.writerow([phase, AV, P])
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
        
        ax.text(.95, .98, plttext, size=INPLOT_LEGEND_FONTSIZE,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        
        # format subplot
        if i%PLOTS_PER_ROW == 0:
            ax.set_title('Phase: {}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel('$E(V-X)$', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_title('{}'.format(phase), fontsize=AXIS_LABEL_FONTSIZE)
    
        plt.xlim(1.0, 3.0)
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
    F.close()
    ############################


################################################################################
### MAIN #######################################################################


def main():
    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
    
    fig = plt.figure(figsize = (20, 12))
    
    
    plot_phase_excesses('SN2012CU', load_12cu_colors, RED_LAW, filters_bucket, zp_bucket)
    
    # format figure
    fig.suptitle('SN2012CU: Color Excess Per Phase', fontsize=TITLE_FONTSIZE)
    
    fig.text(0.5, .05, 'Inverse Wavelength ($1 / \mu m$)',
                fontsize=AXIS_LABEL_FONTSIZE, horizontalalignment='center')
    
    p1, = plt.plot(np.array([]), np.array([]), 'k--')
    p2, = plt.plot(np.array([]), np.array([]), 'r-')
    fig.legend([p1, p2], ['Fitzpatrick-Massa 1999*', 'Power-Law (Goobar 2008)'],
               loc=1, bbox_to_anchor=(0, 0, .97, .99), ncol=2, prop={'size':LEGEND_FONTSIZE})
    
    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.90, wspace=0.2, hspace=0.2)
    plt.show()
    
if __name__=='__main__':
    main()
