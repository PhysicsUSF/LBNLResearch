'''
::Author::
Andrew Stocker

::Description::
This program will fit RV based on color excess of 2012cu

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
from scipy.stats import chisquare


### VARS ###
STEPS = 80
N_BUCKETS = 20

EBV_GUESS = 1.1
EBV_PAD = .3
RV_GUESS = 2.75
RV_PAD = .75


def load_12cu_excess(filters, zp):
        prefix = zp['prefix']
        
        EXCESS = {}
        
        # correct for Milky Way extinction
        sn12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)
        sn12cu = filter(lambda t: t[0]<28, sn12cu)
        phases = [t[0] for t in sn12cu]

        sn12cu_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')/zp['V']) for t in sn12cu]
        sn12cu_colors = {i:{} for i in xrange(len(phases))}
        for f in filters:
                band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)/zp[f]) for t in sn12cu]
                band_colors = np.array(sn12cu_vmags)-np.array(band_mags)
                for i, color in enumerate(band_colors):
                        sn12cu_colors[i][f] = color
        
        sn11fe = l.interpolate_spectra(phases, l.get_11fe())
        for i, phase, sn11fe_phase in izip(xrange(len(phases)), phases, sn11fe):
                sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)/zp[f])
                               for f in filters}
                sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]
                ref_colors = [sn12cu_colors[i][f] for f in filters]
                phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)
                EXCESS[i] = phase_excesses
        
        return EXCESS, phases


def plot_contour(red_law, ref_excess, filter_eff_waves,
                 ebv, ebv_pad, rv, rv_pad, steps, ax=None):
                         
        x = np.linspace(ebv-ebv_pad, ebv+ebv_pad, steps)
        y = np.linspace(rv-rv_pad, rv+rv_pad, steps)
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros( X.shape )
        
        for i, EBV in enumerate(x):
                for j, RV in enumerate(y):
                        ftz_curve = red_law(filter_eff_waves,
                                            np.zeros(filter_eff_waves.shape),
                                            -EBV, RV,
                                            return_excess=True)
                        Z[i,j] = np.sum((ftz_curve-ref_excess)**2)
        
        # find minimum
        ssr_min = np.min(Z)
        mindex = np.where(Z==ssr_min)
        mx, my = mindex[1][0], mindex[0][0]
        
        dof = float(N_BUCKETS-2)  # degrees of freedom
        CHISQ = (dof/ssr_min)*Z   # rescale ssr to be chi-sq; min is now == dof
        
        CDF = 1 - np.exp((-(CHISQ-dof))/2)
        
        # find 1-sigma and 2-sigma errors based on confidence
        maxebv_1sig, maxebv_2sig, minebv_1sig, minebv_2sig = x[mx], x[mx], x[mx], x[mx]
        maxrv_1sig, maxrv_2sig, minrv_1sig, minrv_2sig = y[my], y[my], y[my], y[my]
        for i, EBV in enumerate(x):
                for j, RV in enumerate(y):
                        conf = CDF[j,i]
                        if conf<0.683:
                                maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                                minebv_1sig = np.minimum(minebv_1sig, EBV)
                                maxrv_1sig = np.maximum(maxrv_1sig, RV)
                                minrv_1sig = np.minimum(minrv_1sig, RV)
                        elif conf<0.955:
                                maxebv_2sig = np.maximum(maxebv_2sig, EBV)
                                minebv_2sig = np.minimum(minebv_2sig, EBV)
                                maxrv_2sig = np.maximum(maxrv_2sig, RV)
                                minrv_2sig = np.minimum(minrv_2sig, RV)
        
        if ax != None:
                # plot contours
                contour_levels = [0.683, 0.955]
                plt.contourf(X, Y, CDF, levels=contour_levels)
                C = plt.contour(X, Y, CDF, levels=contour_levels)
                plt.clabel(C, colors='k', fmt='%.2f')
                
                # mark minimum
                plt.scatter(x[mx], y[my], marker='s')
                
                plttext = "BEST:" + \
                          "\n$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                          "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
                          
                plttext = plttext.format(x[mx], maxebv_1sig-x[mx], x[mx]-minebv_1sig,
                                         y[my], maxrv_1sig-y[my], y[my]-minrv_1sig)
                
                ax.text(.05, .95, plttext, size=16,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
                        
                plt.ylim(rv-rv_pad, rv+rv_pad)
                plt.xlim(ebv-ebv_pad, ebv+ebv_pad)
                
        return x, y, CDF, (x[mx], y[my]), (minebv_1sig, maxebv_1sig), \
                                          (minebv_2sig, maxebv_2sig), \
                                          (minrv_1sig,  maxrv_1sig), \
                                          (minrv_2sig,  maxrv_2sig)
        

def main():
        filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
        
        filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff
                                     for f in filters_bucket]
                                    )
        
        sn12cu_excess, phases = load_12cu_excess(filters_bucket, zp_bucket)
        
        
        fig = plt.figure()
        
        for i, phase in enumerate(phases):
                print "Plotting Phase: {}".format(phase)
                
                ax = plt.subplot(2,6,i+1)
                plot_contour(redden_fm, sn12cu_excess[i], filter_eff_waves,
                             EBV_GUESS, EBV_PAD, RV_GUESS, RV_PAD, STEPS, ax)
                
                if i%6 == 0:
                        plt.ylabel('$R_V$')
                if i>=6:
                        plt.xlabel('$E(B-V)$')
                ax.set_title("phase: {}".format(phase))
        
        fig.suptitle('SN2012CU: $E(B-V)$ vs. $R_V$ Contour Plot per Phase', fontsize=18)
        plt.show()
    
        
def get_12cu_best_ebv_rv():
        filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
        
        filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff
                                     for f in filters_bucket]
                                    )
        
        sn12cu_excess, phases = load_12cu_excess(filters_bucket, zp_bucket)
        
        
        SN12CU_CHISQ_DATA = []
        for i, phase in enumerate(phases):
                print "Getting Phase: {}".format(phase)
                
                X, Y, CDF, SSRMIN, ebv_1sig, \
                ebv_2sig, rv_1sig, rv_2sig = plot_contour(redden_fm, sn12cu_excess[i],
                                                          filter_eff_waves, EBV_GUESS,
                                                          EBV_PAD, RV_GUESS, RV_PAD, STEPS
                                                          )
                
                # get best AV and calculate error in quadrature
                BEST_AV = SSRMIN[0]*SSRMIN[1]
                AV_1SIG = (BEST_AV-np.sqrt((ebv_1sig[0]-SSRMIN[0])**2 + (rv_1sig[0]-SSRMIN[1])**2),
                           BEST_AV+np.sqrt((ebv_1sig[1]-SSRMIN[0])**2 + (rv_1sig[1]-SSRMIN[1])**2)
                           )
                AV_2SIG = (BEST_AV-np.sqrt((ebv_2sig[0]-SSRMIN[0])**2 + (rv_2sig[0]-SSRMIN[1])**2),
                           BEST_AV+np.sqrt((ebv_2sig[1]-SSRMIN[0])**2 + (rv_2sig[1]-SSRMIN[1])**2)
                           )
                
                
                SN12CU_CHISQ_DATA.append({'phase'   : phase,
                                          'x'       : X,
                                          'y'       : Y,
                                          'CDF'     : CDF,
                                          'BEST_EBV': SSRMIN[0],
                                          'BEST_RV' : SSRMIN[1],
                                          'BEST_AV' : BEST_AV,
                                          'EBV_1SIG': ebv_1sig,
                                          'EBV_2SIG': ebv_2sig,
                                          'RV_1SIG' : rv_1sig,
                                          'RV_2SIG' : rv_2sig,
                                          'AV_1SIG' : AV_1SIG,
                                          'AV_2SIG' : AV_2SIG
                                          })
        
        return SN12CU_CHISQ_DATA
        
        
'''
To Do:
======
-plot 12cu and 11fe light curves on top of eachother in vein of Periera's plot (possibly with BMAX matching)
-do these plots for power-law
DONE-project errors onto axis for contour plot, export these errors when imported for excess plot
DONE-check ftz curves that on edge of error snake
'''

if __name__ == "__main__":
        main()
