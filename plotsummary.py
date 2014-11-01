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
INPLOT_LEGEND_FONTSIZE = 20
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
        plt.legend(loc=2, ncol=3, prop={'size':LEGEND_FONTSIZE})
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
        
        plt.plot(phases, WAVG*np.ones(phases.shape), color+'--', linewidth=2,
                 label='Average {}={:.2f}$'.format(name[:-1], WAVG))
        #ax.fill_between(phases, stdev_lo, stdev_hi, facecolor=color, alpha=0.3)
        plt.errorbar(phases, data, yerr=[err_lo, err_hi], fmt=color+'s', mec=color, mew=2)
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

        
        fig = plt.figure()
        
        ax = plt.subplot(311)
        plot_err(ax, '$E(B-V)$', phases, EBVS, ebv_err_lo, ebv_err_hi, 'b', TICK_LABEL_FONTSIZE)
        
        ax = plt.subplot(312)
        plot_err(ax, '$R_V$', phases, RVS, rv_err_lo, rv_err_hi, 'r', TICK_LABEL_FONTSIZE)
        
        ax = plt.subplot(313)
        plot_err(ax, '$A_V$', phases, AVS, av_err_lo, av_err_hi, 'g', TICK_LABEL_FONTSIZE)


        plt.xlabel('Days Relative to B-Maximum (MJD {:.1f})'.format(56104.7862735),
                        fontsize=AXIS_LABEL_FONTSIZE, labelpad=AXIS_LABEL_FONTSIZE+4)
        fig.suptitle('SN2012CU: Variation in $E(B-V)$, $R_V$, and $A_V$ over time\n', fontsize=TITLE_FONTSIZE)
        plt.show()
        

if __name__=="__main__":
        main()
