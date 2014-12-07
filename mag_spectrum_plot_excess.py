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
#from mag_spectrum_fitting import filter_features



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
                ind_min = np.abs(red_curve).argmin()
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


## This function really should be imported from mag_spectrum_fitting.py if I can get that to work.
def filter_features(features, wave):
    intersection = np.array([False]*wave.shape[0])
    for feature in features:
        intersection |= ((wave>feature[0])&(wave<feature[1]))
                
    return ~intersection






def main(title, info_dict):
    
    fig = plt.figure(figsize = (20, 12))#, dpi = 10)
    
    obs_SN = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
    phases = [t[0] for t in obs_SN]
    
    pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
    
    
    numrows = (len(phases)-1)//PLOTS_PER_ROW + 1
    pmin, pmax = np.min(phases), np.max(phases)


## To figure out the wavelength of the "zero" of the F99 law.  Using different phases, the value varies a bit (my estimate: 5413.5+/-0.1).
## Haven't looked at this too closely.  But probably because of numerical error, such as during the interpolation process.  It's interesting
## that, reading through the fm_unred.py, which AS and ZR have adopted, it's not obvious what this value should be.
## Here I have arbitrarily chosen the first phase to infer what this wavelength should be.  -XH 11/18/14
    best_ebv = info_dict['ebv'][0]
    best_rv  = info_dict['rv'][0]
    ref_wave = pristine_11fe[0][1].wave
    red_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
    ind_min = np.abs(red_curve).argmin()
    f99wv = np.array([ref_wave[ind_min-1],ref_wave[ind_min],ref_wave[ind_min+1]])
    f99ebv = np.array([red_curve[ind_min-1],red_curve[ind_min],red_curve[ind_min+1]])
    V_wave = np.interp(0., f99ebv, f99wv)
    print 'zero EBV wavelength (V band wavelength):', V_wave  # As I said, this value should be close to 5413.5 A.


    V_wave = 5413.5  # See the comment for the above block of code.
    # AS previously used V_wave = 5417.2


    V_band = [(5412., 5414., 'Vband')]
    del_lamb = 1.
    band_steps = 1200
    V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)
    print V_band_range


    for i, phase in enumerate(phases):

        
        print "Plotting phase {} ...".format(phase)
        ax = plt.subplot(numrows, PLOTS_PER_ROW, i+1)
        
        ref = pristine_11fe[i]
        red = obs_SN[i]
        
        best_ebv = info_dict['ebv'][i]
        best_rv  = info_dict['rv'][i]
        
        ref_wave = ref[1].wave
        ref_flux = ref[1].flux
        ref_var = ref[1].error    # 11fe variance.
        
        ref_interp = interp1d(ref_wave, ref_flux)
        red_interp = interp1d(red[1].wave, red[1].flux)
        
        
        red_flux = red_interp(ref_wave)  # the type of red_interp is <class 'scipy.interpolate.interpolate.interp1d'>, and not just an array.  It probably behaves as a function. One could also do red_interp_flux = interp1d(red_wave, red[1].flux)(ref_wave) -XH Nov 18, 2014
        red_interp_var  = interp1d(red[1].wave, red[1].error)(ref_wave)  # 12cu variance.
 
        Vband_mask = filter_features(V_band, ref_wave) # Not the most efficient way of doing things, but this statement is here because ref_wave is inside the for loop -- also inefficient. Should fix this.

## single wavelength magnitude
        ref_single_wave_mag = (-2.5*np.log10(ref_flux))
        red_single_wave_mag = (-2.5*np.log10(red_flux))


        #excess_ref = (-2.5*np.log10(np.mean(ref_flux))) - (-2.5*np.log10(ref_flux))  # need to add var's as weights.
        #excess_red = (-2.5*np.log10(np.mean(red_flux))) - (-2.5*np.log10(red_flux))  # need to add var's as weights.



        ref_single_V_mag = -2.5*np.log10(ref_interp(V_wave))
        red_single_V_mag = -2.5*np.log10(red_interp(V_wave))

        ref_V_mag = -2.5*np.log10(ref_interp(V_band_range).mean())  # need to add var's as weights.
        red_V_mag = -2.5*np.log10(red_interp(V_band_range).mean())  # need to add var's as weights.

# This way seems to give wrong answer.
#          ref_flux_V_mag = -2.5*np.log10(np.average(ref_flux[Vband_mask]))
#        red_flux_V_mag = -2.5*np.log10(np.average(red_flux[Vband_mask]))

#        color_ref =  ref_single_V_mag - ref_single_wave_mag
#        color_red =  red_single_V_mag - red_single_wave_mag

        color_ref = ref_V_mag - ref_single_wave_mag
        color_red = red_V_mag - red_single_wave_mag


        print '\n\n\n'
        print 'single lambda V band for 11fe', ref_single_V_mag
        print 'V band for 11fe', ref_V_mag
        print 'single lambda V band for 12cu', red_single_V_mag
        print 'V band for 12cu', red_V_mag

        print 'ABS(Avg_mag - V_mag for 11fe) - (Avg_mag - V_mag for 12cu)', np.abs(-2.5*np.log10(ref_interp(V_wave)/np.mean(ref_flux)) - -2.5*np.log10(red_interp(V_wave)/np.mean(red_flux)))
        

        print '\n\n\n'
        
        
        excess = color_red - color_ref
        
        
        # convert effective wavelengths to inverse microns
        ref_wave_inv = 10000./ref_wave
        mfc_color = plt.cm.cool(5./11)    
        
        # plot excess
        plt.plot(ref_wave_inv, excess, '.', color=mfc_color,
                 ms=6, mec='none', mfc=mfc_color, alpha=0.8)
        
        # plot reddening curve
        fm_curve = redden_fm(ref_wave, np.zeros(ref_wave.shape), -best_ebv, best_rv, return_excess=True)
        plt.plot(ref_wave_inv, fm_curve, 'k--')
        
        # plot where V band is.
        plt.plot([ref_wave_inv.min(), ref_wave_inv.max()], [0, 0] ,'--')
        plt.plot([1e4/V_wave, 1e4/V_wave], [fm_curve.min(), fm_curve.max()] ,'--')
        
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
                  "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}"
        
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
    filenm = filter(str.isalnum, title)+'.png' # to get rid of white space and parentheses; and add extension.
    fig.savefig(filenm)

#plt.show()



if __name__=='__main__':
    
    info_dict1 = cPickle.load(open("spectra_mag_fit_results_FILTERED.pkl", 'rb'))
    info_dict2 = cPickle.load(open("spectra_mag_fit_results_UNFILTERED.pkl", 'rb'))
    
    i = 0
    for t in zip(["SN2012cu (Feature Filtered)", "SN2012cu"], [info_dict1, info_dict2]):
        if i > 0: break
        main(t[0], t[1])
        i += 1
