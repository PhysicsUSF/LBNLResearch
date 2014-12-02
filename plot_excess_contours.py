'''
::Author::
Andrew Stocker


::Modidfied by::
Xiaosheng Huang

- I have changed 'vega' to 'ab' in loader.py.  1) to make easier comparison between photometric fit and spectral fit.  2) I ran into an error with 'vega' -- it seems that it can't retrieve data from a ftp site.  It really ought to be changed here and on the command line.

- Even with AB mag, I have recovered low RV values.  The last phase still looks different from other phases.

- At 40 bins or above, one need to remove nan to make it work.  - Dec 2. 2014



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
STEPS = 40
N_BUCKETS = 40

EBV_GUESS = 1.1
EBV_PAD = .3
RV_GUESS = 2.75
RV_PAD = .75


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 20
LEGEND_FONTSIZE = 15


################################################################################

def load_12cu_excess(filters, zp):
        prefix = zp['prefix']  # This specifies units as inverse micron or angstrom; specified in the function call to l.generate_buckets().
        
        print 'filters', filters
        print "zp['V']", zp['V']  # the bands are numbered except V band, which is labeled as 'V'.
        
        
        EXCESS = {}
        
        # correct for Milky Way extinction
        sn12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)
        sn12cu = filter(lambda t: t[0]<28, sn12cu)   # here filter() is a python built-in function.
        phases = [t[0] for t in sn12cu]
        
        #      print 'sn12cu[1]',
        
        ## the method bandflux() returns (bandflux, bandfluxerr), see spectral.py
        ## Further need to understandhow bandflux and bandfluerr are calcuated in spectral.py.
        ## It seems that there may be a conversion between photon flux and energy flux.
        sn12cu_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')[0]/zp['V']) for t in sn12cu]
#        print '\n\n', dir(sn12cu[0][1])
#        print '\n\n', sn12cu[0][1].wave
#        print '\n\n ERROR:', sn12cu[0][1].error
#
#        print sn12cu[0][1].bandflux(prefix+'V')
#        print -2.5*np.log10(sn12cu[0][1].bandflux(prefix+'V')), '\n\n'
#        exit(1)
#        print '\n\n', sn12cu[1][1].bandflux(prefix+'V'), '\n\n'
#        print '\n\n', sn12cu[2][1].bandflux(prefix+'V'), '\n\n'
#        print '\n\n', sn12cu[3][1].bandflux(prefix+'V'), '\n\n'
#        print '\n\n', sn12cu[4][1].bandflux(prefix+'V'), '\n\n'
#        print '\n\n', sn12cu[5][1].bandflux(prefix+'V'), '\n\n'
#
#        print 'sn12cu_vmags', sn12cu_vmags
#        exit(1)
        sn12cu_colors = {i:{} for i in xrange(len(phases))}
        for f in filters:
                band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)[0]/zp[f]) for t in sn12cu]
                band_colors = np.array(sn12cu_vmags)-np.array(band_mags)
                for i, color in enumerate(band_colors):
                        sn12cu_colors[i][f] = color
        
        sn11fe = l.interpolate_spectra(phases, l.get_11fe())
        for i, phase, sn11fe_phase in izip(xrange(len(phases)), phases, sn11fe):
                sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)[0]/zp[f])
                               for f in filters}
                
#                print 'prefix+filters[0]', prefix+filters[0]
#                exit(1)
#                print sn11fe_phase[1].bandflux(prefix+filters[0])
#                print sn11fe_phase[1].bandflux(prefix+filters[-1])
#                exit(1)
#                print -2.5*np.log10( sn11fe_phase[1].bandflux(prefix+filters[0])/zp[filters[0]])
#                exit(1)
#                print 'sn11fe_mag', sn11fe_mags
#                print 'sn11fe_mag length', len(sn11fe_mags)
#                exit(1)
#
                sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters] ## Note: V-band magnitude for 11fe and 12cu are treated differently;
                                                                                   ## need to fix this.  -XH
#                print 'sn11fe_colors', sn11fe_colors
#                print 'sn11fe_colors', len(sn11fe_colors)
#                exit(1)
                ref_colors = [sn12cu_colors[i][f] for f in filters]
#                print 'ref_colors', ref_colors
#                print 'ref_colors', len(ref_colors)
#                exit(1)


                ## why are phases > 20 singled out??  -XH
                if phase>20:
                        print "phase: ",phase
                        
                        print "sn12cu colors:"
                        pprint( zip(filters, ref_colors) )
                        
                        print "sn11fe colors:"
                        pprint( zip(filters, sn11fe_colors) )
                
                phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)
#                print 'np.array(ref_colors).shape', np.array(ref_colors).shape
#                print 'np.array(ref_colors)[:, 0]', np.array(ref_colors)[:, 0]
#                print 'np.array(ref_colors)[:, 1]', np.array(ref_colors)[:, 1]
#
#                exit(1)
                EXCESS[i] = phase_excesses

        
        return EXCESS, phases


################################################################################

def plot_contour(subplot_index, phase, red_law, ref_excess, filter_eff_waves,
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
                        if i==0 and j==0:
                                print "reddening excess:", ftz_curve
                                print "12cu color excess:", ref_excess
#                        print 'ftz_curve.shape', ftz_curve.shape
#                        print 'ref_excess.shape', ref_excess.shape
#                        print 'ftz_curve', ftz_curve
#                        print 'ref_excess', ref_excess
                        #exit(1)
                        #print 'ftz_curve', ftz_curve
                        #print 'ref_excess', ref_excess
                        
                        nanvals = np.isnan(ref_excess)
                        
                        nanmask = ~nanvals
                        Z[j,i] = np.sum((ftz_curve-ref_excess)[nanmask]**2)


        if np.sum(nanvals):
            print '\n\n\nWARNING. WARNGING. WARNTING.'
            print 'WARNING: THERE ARE %d BANDS WITH NAN VALUES.' % (np.sum(nanvals))
            print 'WARNING. WARNGING. WARNTING.\n\n\n'


        #print Z
        
        # find minimum ssr
        ssr_min = np.min(Z)
        mindex = np.where(Z==ssr_min)
        print 'Z', Z
        print 'ssr_min', ssr_min
        print 'mindex', mindex
        print 'mindex length', len(mindex)
        mx, my = mindex[1][0], mindex[0][0]
        
        print "BEST E(B-V): {}".format(x[mx])
        print "BEST RV: {}".format(y[my])
        
        dof = float(N_BUCKETS-1-2)  # degrees of freedom (V-band is fixed, N_BUCKETS-1 floating data pts)
        CHISQ = (dof/ssr_min)*Z   # rescale ssr to be chi-sq; min is now == dof
        chisq_min = np.min(CHISQ)
        
        CDF = 1 - np.exp((-(CHISQ-dof))/2)  # calculation cumulative distribution func
        
        # find 1-sigma and 2-sigma errors based on confidence
        maxebv_1sig, maxebv_2sig, minebv_1sig, minebv_2sig = x[mx], x[mx], x[mx], x[mx]
        maxrv_1sig, maxrv_2sig, minrv_1sig, minrv_2sig = y[my], y[my], y[my], y[my]
        for i, EBV in enumerate(x):
                for j, RV in enumerate(y):
                        #conf = CDF[j,i]
                        _chisq = CHISQ[j,i]-chisq_min
                        #if conf<0.683:
                        if _chisq<1.00:
                                maxebv_1sig = np.maximum(maxebv_1sig, EBV)
                                minebv_1sig = np.minimum(minebv_1sig, EBV)
                                maxrv_1sig = np.maximum(maxrv_1sig, RV)
                                minrv_1sig = np.minimum(minrv_1sig, RV)
                        #elif conf<0.955:
                        elif _chisq<4.00:
                                maxebv_2sig = np.maximum(maxebv_2sig, EBV)
                                minebv_2sig = np.minimum(minebv_2sig, EBV)
                                maxrv_2sig = np.maximum(maxrv_2sig, RV)
                                minrv_2sig = np.minimum(minrv_2sig, RV)
        
        # get best AV and calculate error in quadrature
        best_av = x[mx]*y[my]
        av_1sig = (best_av-np.sqrt((minebv_1sig-x[mx])**2 + (minrv_1sig-y[my])**2),
                   best_av+np.sqrt((maxebv_1sig-x[mx])**2 + (maxrv_1sig-y[my])**2)
                   )
        av_2sig = (best_av-np.sqrt((minebv_2sig-x[mx])**2 + (minrv_2sig-y[my])**2),
                   best_av+np.sqrt((maxebv_2sig-x[mx])**2 + (maxrv_2sig-y[my])**2)
                   )
        
        #print "EBV", minebv_1sig, maxebv_1sig
        #print "RV", minrv_1sig, maxrv_1sig
        
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
                
                #plt.contour(X, Y, CHISQ-chisq_min, levels=[1.0, 4.0], colors=['r', 'g'])
                
                # mark minimum
                plt.scatter(x[mx], y[my], marker='s', facecolors='r')
                
                # show results on plot
                if subplot_index%6==0:
                        plttext1 = "Phase: {}".format(phase)
                else:
                        plttext1 = "{}".format(phase)
                        
                plttext2 = "$E(B-V)={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                           "\n$R_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$" + \
                           "\n$A_V={:.2f}\pm^{{{:.2f}}}_{{{:.2f}}}$"
                plttext2 = plttext2.format(x[mx], maxebv_1sig-x[mx], x[mx]-minebv_1sig,
                                           y[my], maxrv_1sig-y[my], y[my]-minrv_1sig,
                                           best_av, av_1sig[1]-best_av, best_av-av_1sig[0]
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
                        ax.axhspan(2.9, (rv+rv_pad), facecolor='k', alpha=0.1)
                
                else:
                        ax.text(.04, .95, plttext1, size=AXIS_LABEL_FONTSIZE,
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes)
                        ax.text(.04, .32, plttext2, size=INPLOT_LEGEND_FONTSIZE,
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes)
                        ax.axhspan(3.32, (rv+rv_pad), facecolor='k', alpha=0.1)
                        ax.axhspan((rv-rv_pad), 2.5, facecolor='k', alpha=0.1)
                        
                        
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
                labels[0] = labels[-1] = ''
                ax2.set_yticklabels(labels)
                ax2.get_yaxis().set_tick_params(direction='in', pad=-30)
                
                plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
                plt.setp(ax2.get_yticklabels(), fontsize=TICK_LABEL_FONTSIZE)
        
                
        return x, y, CDF, x[mx], y[my], best_av, (minebv_1sig, maxebv_1sig), \
                                                 (minebv_2sig, maxebv_2sig), \
                                                 (minrv_1sig,  maxrv_1sig), \
                                                 (minrv_2sig,  maxrv_2sig), \
                                                 av_1sig, \
                                                 av_2sig

        

def get_all_phases_best_fit():
        red_law = redden_fm
        
        filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
        
        filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff
                                     for f in filters_bucket]
                                    )
        
        sn12cu_excess, phases = load_12cu_excess(filters_bucket, zp_bucket)
        
        
        x = np.linspace(EBV_GUESS-EBV_PAD, EBV_GUESS+EBV_PAD, STEPS)
        y = np.linspace(RV_GUESS-RV_PAD, RV_GUESS+RV_PAD, STEPS)
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros( X.shape )
        
        for i, EBV in enumerate(x):
                for j, RV in enumerate(y):
                        ftz_curve = red_law(filter_eff_waves,
                                            np.zeros(filter_eff_waves.shape),
                                            -EBV, RV,
                                            return_excess=True)
                        
                        ftz_curve_stack = np.hstack(len(phases)*(ftz_curve,))
                        
                        ref_excess_stack = np.hstack(tuple([sn12cu_excess[k] for k in xrange(len(phases))]))
                        
                        Z[j,i] = np.sum((ftz_curve_stack-ref_excess_stack)**2)
        
        
        # find minimum ssr
        ssr_min = np.min(Z)
        mindex = np.where(Z==ssr_min)
        mx, my = mindex[1][0], mindex[0][0]
        
        n_dataPts = 12*(N_BUCKETS-1)
        dof = float(n_dataPts-2)  # degrees of freedom (V-band is fixed, N_BUCKETS-1 floating data pts -2 fitting parameters)
        CHISQ = (dof/ssr_min)*Z   # rescale ssr to be chi-sq; min is now == dof
        
        CDF = 1 - np.exp((-(CHISQ-dof))/2)  # calculation cumulative distribution func
        
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
        
        # get best AV and calculate error in quadrature
        best_av = x[mx]*y[my]
        av_1sig = (best_av-np.sqrt((minebv_1sig-x[mx])**2 + (minrv_1sig-y[my])**2),
                   best_av+np.sqrt((maxebv_1sig-x[mx])**2 + (maxrv_1sig-y[my])**2)
                   )
        av_2sig = (best_av-np.sqrt((minebv_2sig-x[mx])**2 + (minrv_2sig-y[my])**2),
                   best_av+np.sqrt((maxebv_2sig-x[mx])**2 + (maxrv_2sig-y[my])**2)
                   )
        
        
        template = "{} = {:.4f}; 1SIGMA: (+{:.4f}/-{:.4f}); 2SIGMA: (+{:.4f}/-{:.4f})"
        print template.format("E(B-V)", x[mx], x[mx]-minebv_1sig, maxebv_1sig-x[mx],
                                        x[mx]-minebv_2sig, maxebv_2sig-x[mx])
        print template.format("RV", y[my], y[my]-minrv_1sig,  maxrv_1sig-y[my],
                                        y[my]-minrv_2sig,  maxrv_2sig-y[my])
        print template.format("AV", best_av, best_av-av_1sig[0], av_1sig[1]-best_av,
                                        best_av-av_2sig[0], av_2sig[1]-best_av)
        

################################################################################
        
def get_12cu_best_ebv_rv(red_law, filters, zp):
        filter_eff_waves = np.array([snc.get_bandpass(zp['prefix']+f).wave_eff
                                     for f in filters])
        
        sn12cu_excess, phases = load_12cu_excess(filters, zp)
        
        
        SN12CU_CHISQ_DATA = []
        for i, phase in enumerate(phases):
                print "Getting phase {} ...".format(phase)
                
                x, y, CDF, \
                best_ebv, best_rv, best_av, \
                ebv_1sig, ebv_2sig, \
                rv_1sig, rv_2sig, \
                av_1sig, av_2sig = plot_contour(i, phase, red_law, sn12cu_excess[i],
                                                filter_eff_waves, EBV_GUESS,
                                                EBV_PAD, RV_GUESS, RV_PAD, STEPS
                                                )
                
                SN12CU_CHISQ_DATA.append({'phase'   : phase,
                                          'x'       : x,
                                          'y'       : y,
                                          'CDF'     : CDF,
                                          'BEST_EBV': best_ebv,
                                          'BEST_RV' : best_rv,
                                          'BEST_AV' : best_av,
                                          'EBV_1SIG': ebv_1sig,
                                          'EBV_2SIG': ebv_2sig,
                                          'RV_1SIG' : rv_1sig,
                                          'RV_2SIG' : rv_2sig,
                                          'AV_1SIG' : av_1sig,
                                          'AV_2SIG' : av_2sig
                                          })
        
        return SN12CU_CHISQ_DATA



################################################################################

if __name__ == "__main__":

    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
    
    #        print 'filters_bucket', filters_bucket
    #        print 'zp_bucket', zp_bucket
    #        exit(1)
    
    
    filter_eff_waves = np.array([snc.get_bandpass(zp_bucket['prefix']+f).wave_eff
                                 for f in filters_bucket])
        
    sn12cu_excess, phases = load_12cu_excess(filters_bucket, zp_bucket)
    
    
    fig = plt.figure(figsize = (20, 12))
    
    for i, phase in enumerate(phases):
        print "Plotting phase {} ...".format(phase)
            
        ax = plt.subplot(2,6,i+1)
                #                print 'sn12cu_excess[i].shape', sn12cu_excess[i].shape
                #                exit(1)
        plot_contour(i, phase, redden_fm, sn12cu_excess[i], filter_eff_waves,
                             EBV_GUESS, EBV_PAD, RV_GUESS, RV_PAD, STEPS, ax)
            

    fig.subplots_adjust(left=0.04, bottom=0.08, right=0.95, top=0.92, hspace=.06, wspace=.1)
    fig.suptitle('SN2012CU: $E(B-V)$ vs. $R_V$ Contour Plot per Phase', fontsize=TITLE_FONTSIZE)
    plt.show()

