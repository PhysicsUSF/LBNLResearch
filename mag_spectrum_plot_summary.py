import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cPickle

from copy import deepcopy


TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 16
LEGEND_FONTSIZE = 20


def get_errors(best_ebvs, best_rvs, x, y, chi2s, chi2_reductions):
        l = best_ebvs.shape[0]
        
        maxebv_1sig, minebv_1sig = deepcopy(best_ebvs), deepcopy(best_ebvs)
        maxrv_1sig, minrv_1sig = deepcopy(best_rvs), deepcopy(best_rvs)
        
        for i in xrange(l):
                CHI2 = chi2s[i]/chi2_reductions[i]
                CHI2 -= np.min(CHI2)
                
                for e, EBV in enumerate(x):
                        for r, RV in enumerate(y):
                                _chi2 = CHI2[r,e]
                                if _chi2<1.00:
                                        maxebv_1sig[i] = np.maximum(maxebv_1sig[i], EBV)
                                        minebv_1sig[i] = np.minimum(minebv_1sig[i], EBV)
                                        maxrv_1sig[i] = np.maximum(maxrv_1sig[i], RV)
                                        minrv_1sig[i] = np.minimum(minrv_1sig[i], RV)
                                
        return {'ebv': (best_ebvs-minebv_1sig, maxebv_1sig-best_ebvs),
                'rv': (best_rvs-minrv_1sig, maxrv_1sig-best_rvs)}


def main(title, info_dict):
        
        phases = info_dict['phases']
        
        best_ebvs = info_dict['ebv']
        best_rvs = info_dict['rv']
        
        chi2s = info_dict['chi2']
        chi2_reductions = info_dict['chi2_reductions']
        
        x = info_dict['x']
        y = info_dict['y']
        
        errors = get_errors(np.array(best_ebvs), np.array(best_rvs), x, y, chi2s, chi2_reductions)
        
        
        
        fig = plt.figure(figsize=(14, 6))
        
        
        
        ax = plt.subplot(211)
        
        plt.errorbar(phases, best_ebvs, yerr=[errors['ebv'][0], errors['ebv'][1]], fmt='bs', ms=8)
        
        for p in np.arange(-5, 30, 5):
                ax.axvline(p, color='k', linewidth=2, linestyle=':', alpha=0.6)
        for p in np.linspace(.95, 1.15, 5):
                ax.axhline(p, color='k', linewidth=2, linestyle=':', alpha=0.6)
                
        labels = ax.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_yticklabels(labels)
        plt.ylabel("$E(B-V)$", fontsize=20, rotation='horizontal', labelpad=14)
        ax.set_xticklabels([])
        
        
        
        ax = plt.subplot(212)
        
        plt.errorbar(phases, best_rvs, yerr=[errors['rv'][0], errors['rv'][1]], fmt='rs', ms=8)
        
        for p in np.arange(-5,30,5):
                ax.axvline(p, color='k', linewidth=2, linestyle=':', alpha=0.6)
        for p in np.linspace(2.8, 3.4, 4):
                ax.axhline(p, color='k', linewidth=2, linestyle=':', alpha=0.6)
                
        labels = ax.get_yticks().tolist()
        labels[0] = labels[-1] = ''
        ax.set_yticklabels(labels)
        plt.ylabel("$R_V$", fontsize=20, rotation='horizontal', labelpad=14)
        
        
        
        fig.subplots_adjust(hspace=0)
        fig.suptitle('{}: Variation in $E(B-V)$, $R_V$, and $A_V$ over time\n'.format(title), fontsize=TITLE_FONTSIZE)
        plt.show()








if __name__=="__main__":
        
        info_dict1 = cPickle.load(open("spectra_mag_fit_results_FILTERED.pkl", 'rb'))
        info_dict2 = cPickle.load(open("spectra_mag_fit_results_UNFILTERED.pkl", 'rb'))
    
        for t in zip(["SN2012cu (Feature Filtered)", "SN2012cu"], [info_dict1, info_dict2]):
                main(t[0], t[1])
