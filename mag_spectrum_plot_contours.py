import cPickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from pprint import pprint


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15



def plot_contour(subplot_index, phase, info_dict,
                 ebv, ebv_pad, rv, rv_pad, ax=None):
        
        
        ret_phase = info_dict['phases'][subplot_index]
        CHI2 = info_dict['chi2'][subplot_index]
        CHI2_reduction = info_dict['chi2_reductions'][subplot_index]
        x = info_dict['x']
        y = info_dict['y']
        X = info_dict['X']
        Y = info_dict['Y']
        
        CHI2 = CHI2/CHI2_reduction
        chi2_min = np.min(CHI2)
        
        mindex = np.where(CHI2==chi2_min)
        mx, my = mindex[1][0], mindex[0][0]
        
        print "BEST E(B-V): {}".format(x[mx])
        print "BEST RV: {}".format(y[my])
        
        
        
        CDF = 1 - np.exp((-(CHI2-chi2_min))/2)  # calculation cumulative distribution func
        
        # find 1-sigma and 2-sigma errors based on confidence
        maxebv_1sig, maxebv_2sig, minebv_1sig, minebv_2sig = x[mx], x[mx], x[mx], x[mx]
        maxrv_1sig, maxrv_2sig, minrv_1sig, minrv_2sig = y[my], y[my], y[my], y[my]
        for i, EBV in enumerate(x):
                for j, RV in enumerate(y):
                        _chi2 = CHI2[j,i]-chi2_min
                        if _chi2<1.00:
                                maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                                minebv_1sig = np.minimum(minebv_1sig, EBV)
                                maxrv_1sig = np.maximum(maxrv_1sig, RV)
                                minrv_1sig = np.minimum(minrv_1sig, RV)
                        elif _chi2<4.00:
                                maxebv_2sig = np.maximum(maxebv_2sig, EBV)
                                minebv_2sig = np.minimum(minebv_2sig, EBV)
                                maxrv_2sig = np.maximum(maxrv_2sig, RV)
                                minrv_2sig = np.minimum(minrv_2sig, RV)
        
        #ax.axvline(minebv_1sig, color='r')
        #ax.axvline(maxebv_1sig, color='r')
        #ax.axhline(minrv_1sig, color='r')
        #ax.axhline(maxrv_1sig, color='r')
        
        #ax.axvline(minebv_2sig, color='g')
        #ax.axvline(maxebv_2sig, color='g')
        #ax.axhline(minrv_2sig, color='g')
        #ax.axhline(maxrv_2sig, color='g')
        
        if ax != None:
                # plot contours
                contour_levels = [0.0, 0.683, 0.955, 1.0]
                plt.contourf(X, Y, 1-CDF, levels=[1-l for l in contour_levels], cmap=mpl.cm.summer, alpha=0.5)
                C1 = plt.contour(X, Y, CDF, levels=[contour_levels[1]], linewidths=1, colors=['k'], alpha=0.7)
                
                # mark minimum
                plt.scatter(x[mx], y[my], marker='s', facecolors='r')
                
                # show results on plot
                if subplot_index%6==0:
                        plttext1 = "Phase: {}".format(phase)
                else:
                        plttext1 = "{}".format(phase)
                
                plttext2 = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                           "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
                plttext2 = plttext2.format(x[mx], maxebv_1sig-x[mx], x[mx]-minebv_1sig,
                                           y[my], maxrv_1sig-y[my], y[my]-minrv_1sig,
                                           )
                
                if phase not in [11.5, 16.5, 18.5, 21.5]:
                        ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes)
                        ax.text(.04, .85, plttext2, size=INPLOT_LEGEND_FONTSIZE,
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes)
                        ax.axhspan(3.2, (rv+rv_pad), facecolor='k', alpha=0.1)
                
                else:
                        ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes)
                        ax.text(.04, .26, plttext2, size=INPLOT_LEGEND_FONTSIZE,
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes)
                        ax.axhspan(3.42, (rv+rv_pad), facecolor='k', alpha=0.1)
                        ax.axhspan((rv-rv_pad), 2.75, facecolor='k', alpha=0.1)
                        
                        
                # format subplot...
                plt.ylim(rv-rv_pad, rv+rv_pad)
                plt.xlim(ebv-ebv_pad, ebv+ebv_pad)
                
                ax.set_yticklabels([])
                ax2 = ax.twinx()
                ax2.set_xlim(ebv-ebv_pad, ebv+ebv_pad)
                ax2.set_ylim(rv-rv_pad, rv+rv_pad)
                
                if subplot_index%6 == 5:
                        ax2.set_ylabel('\n$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=5)
                if subplot_index%6 == 0:
                        ax.set_ylabel('$R_V$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=-2)
                if subplot_index>=6:
                        ax.set_xlabel('\n$E(B-V)$', fontsize=AXIS_LABEL_FONTSIZE)
                
                # format x labels
                labels = ax.get_xticks().tolist()
                labels[0] = labels[-1] = ''
                ax.set_xticklabels(labels)
                ax.get_xaxis().set_tick_params(direction='in', pad=-20)
                
                # format y labels
                labels = ax2.get_yticks().tolist()
                labels[0] = labels[1] = labels[-1] = ''
                ax2.set_yticklabels(labels)
                ax2.get_yaxis().set_tick_params(direction='in', pad=-30)
                
                plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
                plt.setp(ax2.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)



        
################################################################################

def allphase_fit(info_dict):
        #### find best fit for all phases
        CHI2_total = np.sum(info_dict['chi2'], axis=0)
        CHI2_total_reduction = np.sum(info_dict['chi2_reductions'])
        
        CHI2_total /= CHI2_total_reduction
        
        chi2_total_min = np.min(CHI2_total)
        
        x = info_dict['x']
        y = info_dict['y']
        mindex = np.where(CHI2_total==chi2_total_min)
        mx, my = mindex[1][0], mindex[0][0]
        
        
        # find 1-sigma and 2-sigma errors based on confidence
        maxebv_1sig, maxebv_2sig, minebv_1sig, minebv_2sig = x[mx], x[mx], x[mx], x[mx]
        maxrv_1sig, maxrv_2sig, minrv_1sig, minrv_2sig = y[my], y[my], y[my], y[my]
        for i, EBV in enumerate(x):
                for j, RV in enumerate(y):
                        _chi2 = CHI2_total[j,i]-chi2_total_min
                        if _chi2<1.00:
                                maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                                minebv_1sig = np.minimum(minebv_1sig, EBV)
                                maxrv_1sig = np.maximum(maxrv_1sig, RV)
                                minrv_1sig = np.minimum(minrv_1sig, RV)
                        elif _chi2<4.00:
                                maxebv_2sig = np.maximum(maxebv_2sig, EBV)
                                minebv_2sig = np.minimum(minebv_2sig, EBV)
                                maxrv_2sig = np.maximum(maxrv_2sig, RV)
                                minrv_2sig = np.minimum(minrv_2sig, RV)
        
        
        print "BEST TOTAL E(B-V): {:.4} (+{:.4}/-{:.4})".format(x[mx], maxebv_1sig-x[mx], x[mx]-minebv_1sig)
        print "BEST TOTAL RV: {:.4} (+{:.4}/-{:.4})".format(y[my], maxrv_1sig-y[my], y[my]-minrv_1sig)
        print "###################################"
        

################################################################################


def main():
        ebv_guess = 1.05
        ebv_pad = 0.25
        
        rv_guess = 3.0
        rv_pad = 0.6
        
        
        info_dict = cPickle.load(open("spectra_mag_fit_results_03-44-32-11-14-2014.pkl", 'rb'))
        phases = info_dict['phases']
        
        allphase_fit(info_dict)
        
        fig = plt.figure()
        
        for i, phase in enumerate(phases):
                print "Plotting phase {} ...".format(phase)
                
                ax = plt.subplot(2,6,i+1)
                
                plot_contour(i, phase, info_dict, ebv_guess, ebv_pad, rv_guess, rv_pad, ax)
        
        
        fig.subplots_adjust(left=0.04, bottom=0.08, right=0.95, top=0.92, hspace=.06, wspace=.1)
        fig.suptitle('SN2012CU: $E(B-V)$ vs. $R_V$ per Phase', fontsize=TITLE_FONTSIZE)
        plt.show()





if __name__=="__main__":
        main()
