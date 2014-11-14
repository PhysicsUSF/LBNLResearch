'''
plots the summary of EBV, RV, and AV changing over time
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
from itertools import izip

TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 16
LEGEND_FONTSIZE = 20


def format_subplot(name, ax, size):
        for p in np.arange(-5,30,5):
                ax.axvline(p, color='k', linewidth=2, linestyle=':', alpha=0.6)
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(size)
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(size)
        labels = ax.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_yticklabels(labels)
        plt.ylabel(name, fontsize=20, rotation='horizontal', labelpad=14)


def wavg(data, los, his):
        quads = np.sqrt(los**2 + his**2)
        notnormalized = np.sum([x*quad for x, quad in izip(data, quads)])
        return notnormalized/np.sum(quads)


def plot_err(ax, name, phases, data, err_lo, err_hi, color, size):
        WAVG = wavg(data, err_lo, err_hi)
        #stdev = np.sqrt(np.var(data))
        #stdev_lo = WAVG-stdev*np.ones(phases.shape)
        #stdev_hi = WAVG+stdev*np.ones(phases.shape)
        
        plt.plot(phases, WAVG*np.ones(phases.shape), color+'--', linewidth=2)
        if name=='$E(B-V)$':
                plt.text(-1, 1.09, '{:.3} (Avg. {} for color excess fit)'.format(WAVG, name),
                color=color, fontsize=TICK_LABEL_FONTSIZE)
        elif name=='$R_V$':
                plt.text(0.2, 2.4, '{:.3} (Avg. {} for color excess fit)'.format(WAVG, name),
                color=color, fontsize=TICK_LABEL_FONTSIZE)
        elif name=='$A_V$':
                plt.text(1, 2.6, '{:.3} (Avg. {} for color excess fit)'.format(WAVG, name),
                color=color, fontsize=TICK_LABEL_FONTSIZE)
        #ax.fill_between(phases, stdev_lo, stdev_hi, facecolor=color, alpha=0.3)
        plt.errorbar(phases, data, yerr=[err_lo, err_hi], fmt=color+'s', ms=8, mfc='w', mec=color, mew=3)
        format_subplot(name, ax, size)



def main():
        SN12CU_CHISQ_DATA = pickle.load(open('sn12cu_chisq_data.pkl', 'rb'))

        phases = np.array([d['phase'] for d in SN12CU_CHISQ_DATA])
        
        EBVS = [d['BEST_EBV'] for d in SN12CU_CHISQ_DATA]
        RVS = [d['BEST_RV'] for d in SN12CU_CHISQ_DATA]
        AVS = [d['BEST_AV'] for d in SN12CU_CHISQ_DATA]
        
        ebv_err_lo = np.array([abs(d['BEST_EBV']-d['EBV_1SIG'][0]) for d in SN12CU_CHISQ_DATA])
        ebv_err_hi = np.array([abs(d['BEST_EBV']-d['EBV_1SIG'][1]) for d in SN12CU_CHISQ_DATA])
        
        rv_err_lo = np.array([abs(d['BEST_RV']-d['RV_1SIG'][0]) for d in SN12CU_CHISQ_DATA])
        rv_err_hi = np.array([abs(d['BEST_RV']-d['RV_1SIG'][1]) for d in SN12CU_CHISQ_DATA])
        
        av_err_lo = np.array([abs(d['BEST_AV']-d['AV_1SIG'][0]) for d in SN12CU_CHISQ_DATA])
        av_err_hi = np.array([abs(d['BEST_AV']-d['AV_1SIG'][1]) for d in SN12CU_CHISQ_DATA])

        
        # load spectra fit results
        results = pickle.load(open("spectra_fit_results_{}iter.pkl".format(200), 'rb'))
        print results
        
        fit_results_kwargs = {'ms':10, 'mfc':'w', 'mew':3}
        
        fig = plt.figure()
        
        ax = plt.subplot(311)
        plot_err(ax, '$E(B-V)$', phases, EBVS, ebv_err_lo, ebv_err_hi, 'b', TICK_LABEL_FONTSIZE)
        plt.plot(phases, results['ebv'], '^', mec='k', **fit_results_kwargs)
        plt.plot(phases, np.average(results['ebv'])*np.ones(phases.shape), 'k--', linewidth=2)
        plt.text(-3, 0.98, '{:.3} (Avg. {} for spectrum comparison fit)'.format(np.average(results['ebv']), '$E(B-V)$'),
                color='k', fontsize=TICK_LABEL_FONTSIZE)
                
        ax = plt.subplot(312)
        plot_err(ax, '$R_V$', phases, RVS, rv_err_lo, rv_err_hi, 'r', TICK_LABEL_FONTSIZE)
        plt.plot(phases, results['rv'], '^', mec='k', **fit_results_kwargs)
        plt.plot(phases, np.average(results['rv'])*np.ones(phases.shape), 'k--', linewidth=2)
        plt.text(-2, 2.93, '{:.3} (Avg. {} for spectrum comparison fit)'.format(np.average(results['rv']), '$R_V$'),
                color='k', fontsize=TICK_LABEL_FONTSIZE)
                
        ax = plt.subplot(313)
        plot_err(ax, '$A_V$', phases, AVS, av_err_lo, av_err_hi, 'g', TICK_LABEL_FONTSIZE)
        plt.plot(phases, results['av'], '^', mec='k', **fit_results_kwargs)
        plt.plot(phases, np.average(results['av'])*np.ones(phases.shape), 'k--', linewidth=2)
        plt.text(-4.5, 2.93, '{:.3} (Avg. {} for spectrum comparison fit)'.format(np.average(results['av']), '$A_V$'),
                color='k', fontsize=TICK_LABEL_FONTSIZE)
                
        plt.xlabel('Days Relative to B-Maximum (MJD {:.1f})'.format(56104.7862735),
                        fontsize=AXIS_LABEL_FONTSIZE, labelpad=AXIS_LABEL_FONTSIZE+4)
        fig.suptitle('SN2012CU: Variation in $E(B-V)$, $R_V$, and $A_V$ over time\n', fontsize=TITLE_FONTSIZE)
        
        plt.subplots_adjust(left=0.09, bottom=0.11, right=0.92, top=0.87)
        p1, = plt.plot(np.array([]), np.array([]), 'k^--', ms=10, mfc='w', mew=2)
        p2, = plt.plot(np.array([]), np.array([]), 'bs--', ms=10, mfc='w', mec='b', mew=2)
        p3, = plt.plot(np.array([]), np.array([]), 'rs--', ms=10, mfc='w', mec='r', mew=2)
        p4, = plt.plot(np.array([]), np.array([]), 'bs--', ms=10, mfc='w', mec='g', mew=2)
        fig.legend([p1, p2, p3, p4], ['Spectrum Comparison Results', 'Color Excess: $E(B-V)$', 'Color Excess: $R_V$', 'Color Excess: $A_V$'],
               loc=9, bbox_to_anchor=[0,0,1,.94], ncol=4, prop={'size':LEGEND_FONTSIZE})
        
        plt.show()
        

if __name__=="__main__":
        main()
