'''
plots the summary of EBV, RV, and AV changing over time
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main():
        SN12CU_CHISQ_DATA = pickle.load(open('sn12cu_chisq_data.pkl', 'rb'))
        FILL_ALPHA = 0.3

        phases = [d['phase'] for d in SN12CU_CHISQ_DATA]
        
        EBVS = [d['BEST_EBV'] for d in SN12CU_CHISQ_DATA]
        RVS = [d['BEST_RV'] for d in SN12CU_CHISQ_DATA]
        AVS = [d['BEST_AV'] for d in SN12CU_CHISQ_DATA]
        
        ebv_err_lo = np.array([abs(d['BEST_EBV']-d['EBV_1SIG'][0]) for d in SN12CU_CHISQ_DATA])
        ebv_err_hi = np.array([abs(d['BEST_EBV']-d['EBV_1SIG'][1]) for d in SN12CU_CHISQ_DATA])
        
        rv_err_lo = np.array([abs(d['BEST_RV']-d['RV_1SIG'][0]) for d in SN12CU_CHISQ_DATA])
        rv_err_hi = np.array([abs(d['BEST_RV']-d['RV_1SIG'][1]) for d in SN12CU_CHISQ_DATA])
        
        av_err_lo = np.array([abs(d['BEST_AV']-d['AV_1SIG'][0]) for d in SN12CU_CHISQ_DATA])
        av_err_hi = np.array([abs(d['BEST_AV']-d['AV_1SIG'][1]) for d in SN12CU_CHISQ_DATA])
        
        fig, ax = plt.subplots(1)
        
        plt.errorbar(phases, EBVS, yerr=[ebv_err_lo, ebv_err_hi], fmt='bs--', mec='b', mew=2, label='$E(B-V)$')
        ax.fill_between(phases, EBVS-ebv_err_lo, EBVS+ebv_err_hi, facecolor='b', alpha=FILL_ALPHA)
        
        plt.errorbar(phases, RVS, yerr=[rv_err_lo, rv_err_hi], fmt='rs--', mec='r', mew=2, label='$R_V$')
        ax.fill_between(phases, RVS-rv_err_lo, RVS+rv_err_hi, facecolor='r', alpha=FILL_ALPHA)
        
        plt.errorbar(phases, AVS, yerr=[av_err_lo, av_err_hi], fmt='gs--', mec='g', mew=2, label='$A_V$')
        ax.fill_between(phases, AVS-av_err_lo, AVS+av_err_hi, facecolor='g', alpha=FILL_ALPHA)
        
        plt.legend(ncol=3)
        plt.xlabel('Phase Relative to B-Maximum')
        plt.title('SN2012CU: Variation in $E(B-V)$, $R_V$, and $A_V$ over time')
        plt.show()
        

if __name__=="__main__":
        main()
