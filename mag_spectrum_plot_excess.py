'''
::Author::
Andrew Stocker

::Description::
This program will plot the color excess plot for 14J and 12CU with various reddening laws.

::Last Modified::
08/01/2014

'''
import cPickle
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




# config
PLOTS_PER_ROW = 6
N_BUCKETS = 20
RED_LAW = redden_fm

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15


def plot_snake(ax, rng, init, red_law, x, y, CHI2, plot2sig=False):
    snake_hi_1sig = deepcopy(init)
    snake_lo_1sig = deepcopy(init)
    if plot2sig:
        snake_hi_2sig = deepcopy(init)
        snake_lo_2sig = deepcopy(init)
    
    for i, EBV in enumerate(x):
        for j, RV in enumerate(y):
            _chi2 = CHI2[j,i]
            if _chi2<1.00:
                red_curve = red_law(rng, np.zeros(rng.shape), -EBV, RV, return_excess=True)
                snake_hi_1sig = np.maximum(snake_hi_1sig, red_curve)
                snake_lo_1sig = np.minimum(snake_lo_1sig, red_curve)
            elif plot2sig and _chi2<4.00:
                red_curve = red_law(rng, np.zeros(rng.shape), -EBV, RV, return_excess=True)
                snake_hi_2sig = np.maximum(snake_hi_2sig, red_curve)
                snake_lo_2sig = np.minimum(snake_lo_2sig, red_curve)
    
    ax.fill_between(10000./rng, snake_lo_1sig, snake_hi_1sig, facecolor='black', alpha=0.3)
    if plot2sig:
        ax.fill_between(10000./rng, snake_lo_2sig, snake_hi_2sig, facecolor='black', alpha=0.1)
        
    return interp1d(rng, snake_lo_1sig), interp1d(rng, snake_hi_1sig)



    


def main(title, info_dict):
    
    fig = plt.figure()
    
    pristine_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in pristine_12cu]
    
    pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    pmin, pmax = np.min(phases), np.max(phases)
    
    for i, phase in enumerate(phases):
        print "Plotting phase {} ...".format(phase)
        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        
        ref = pristine_11fe[i]
        red = pristine_12cu[i]
        
        best_ebv = info_dict['ebv'][i]
        best_rv  = info_dict['rv'][i]
        
        ref_wave = ref[1].wave
        ref_flux = ref[1].flux
        
        ref_interp = interp1d(ref_wave, ref_flux)
        red_interp = interp1d(red[1].wave, red[1].flux)
        
        red_flux = red_interp(ref_wave)
        
        excess_ref = (-2.5*np.log10(ref_interp(5417.2))) - (-2.5*np.log10(ref_flux))
        excess_red = (-2.5*np.log10(red_interp(5417.2))) - (-2.5*np.log10(red_flux))
        
        excess = excess_red - excess_ref
        
        
        # convert effective wavelengths to inverse microns
        ref_wave_inv = 10000./ref_wave
        mfc_color = plt.cm.cool(5./11)    
        
        # plot excess
        plt.plot(ref_wave_inv, excess, '.', color=mfc_color,
                 ms=6, mec='none', mfc=mfc_color, alpha=0.8)
        
        # plot reddening curve
        fm_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        plt.plot(ref_wave_inv, fm_curve, 'k--')
        
        # plot error snake
        x = info_dict['x']
        y = info_dict['y']
        CHI2 = info_dict['chi2'][i]
        CHI2_reduction = info_dict['chi2_reductions'][i]
        CHI2 /= CHI2_reduction
        CHI2 = CHI2 - np.min(CHI2)
        
        slo, shi = plot_snake(ax, ref_wave, fm_curve, redden_fm, x, y, CHI2)
        
        # plot power law reddening curve
        pl_red_curve = redden_pl2(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        plt.plot(ref_wave_inv, pl_red_curve, 'r-')
        
        # find 1-sigma and 2-sigma errors based on confidence
        maxebv_1sig, minebv_1sig = best_ebv, best_ebv
        maxrv_1sig, minrv_1sig = best_rv, best_rv
        for e, EBV in enumerate(x):
                for r, RV in enumerate(y):
                        _chi2 = CHI2[r,e]
                        if _chi2<1.00:
                                maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                                minebv_1sig = np.minimum(minebv_1sig, EBV)
                                maxrv_1sig = np.maximum(maxrv_1sig, RV)
                                minrv_1sig = np.minimum(minrv_1sig, RV)
        
        
        
        ### FORMAT SUBPLOT ###
        
        # print data on subplot
        plttext = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                  "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
        
        plttext = plttext.format(best_ebv, maxebv_1sig-best_ebv, best_ebv-minebv_1sig,
                                 best_rv, maxrv_1sig-best_rv, best_rv-minrv_1sig
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
    
    
    # format figure
    fig.suptitle('{}: Color Excess'.format(title), fontsize=TITLE_FONTSIZE)
    
    fig.text(0.5, .05, 'Inverse Wavelength ($1 / \mu m$)',
                fontsize=AXIS_LABEL_FONTSIZE, horizontalalignment='center')
    
    p1, = plt.plot(np.array([]), np.array([]), 'k--')
    p2, = plt.plot(np.array([]), np.array([]), 'r-')
    fig.legend([p1, p2], ['Fitzpatrick-Massa 1999*', 'Power-Law (Goobar 2008)'],
               loc=1, bbox_to_anchor=(0, 0, .97, .99), ncol=2, prop={'size':LEGEND_FONTSIZE})
    
    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.90, wspace=0.2, hspace=0.2)
    plt.show()



if __name__=='__main__':
    
    info_dict1 = cPickle.load(open("spectra_mag_fit_results_FILTERED.pkl", 'rb'))
    info_dict2 = cPickle.load(open("spectra_mag_fit_results_UNFILTERED.pkl", 'rb'))
    
    for t in zip(["SN2012cu (Feature Filtered)", "SN2012cu"], [info_dict1, info_dict2]):
        main(t[0], t[1])
