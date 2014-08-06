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
STEPS = 20
N_BUCKETS = 20



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


def plot_contour(red_law, ref_excess, filter_eff_waves, ebv, ebv_pad, rv, rv_pad, steps):
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
        
        CDF = 1 - np.exp((-(Z-ssr_min)**2)/2)
        
        contour_levels = [0.68, 0.95]
        plt.contourf(X, Y, CDF, levels=contour_levels)
        C = plt.contour(X, Y, CDF, levels=contour_levels)
        plt.clabel(C, colors='k', fmt='%.2f')
        
        # mark minimum
        plt.scatter(x[mx], y[my], marker='s')
        plt.text(x[mx]-0.08, y[my]-0.08, "({:.2f}, {:.2f})".format(x[mx], y[my]))
        
        return (x[mx], y[my])
        

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
                info = plot_contour(redden_fm, sn12cu_excess[i], filter_eff_waves,
                                    1.1, 0.3, 2.75, 0.75, STEPS)
                
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
        
        
        EBVS, RVS = [], []
        for i, phase in enumerate(phases):
                print "Getting Phase: {}".format(phase)
                
                info = plot_contour(redden_fm, sn12cu_excess[i], filter_eff_waves,
                                    1.1, 0.3, 2.75, 0.75, STEPS)
                
                EBVS.append(info[0])
                RVS.append(info[1])
        
        return phases, EBVS, RVS
        
        
'''
To Do:
======
-email this list to Xiaosheng
-plot 12cu and 11fe light curves on top of eachother in vein of Periera's plot (possibly with BMAX matching)
-show color curves (with same phases as phase plots)
-proper error computation for contour plots using degrees of freedom
-plot proper confidence ellipses
-show best chi-sq on excess by phase fit
-do "color snake" based on 1-sigma ebvs/rvs on contour plots
        -also include best-fit for power-law plot
        
'''

if __name__ == "__main__":
        main()
