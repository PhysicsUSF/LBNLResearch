'''
::Author::
Andrew Stocker

::Description::
This program will plot the color excess plot for 14J and 12CU with various reddening laws.

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
from scipy.interpolate import interp1d
from sys import argv



################################################################################
### VARS #######################################################################

ARGSLEN = len(argv)
PLOT_PHASES = False

## vars for phase plot ##
FONT_SIZE = 14
N_BUCKETS = 20
RED_TYPE = 'fm'
PLOT_OTHER = False
PLOT_ERROR_1 = False
PLOT_ERROR_2 = False
ERROR = 0.3

if ARGSLEN > 1:
    PLOT_PHASES = True
    RED_TYPE = argv[1]
if ARGSLEN > 2:
    try:
        ERROR = float(argv[2])
        PLOT_ERROR_1 = True
    except ValueError:
        PLOT_OTHER = True
if ARGSLEN > 3:
    ERROR = float(argv[3])
    PLOT_ERROR_1 = True
    PLOT_ERROR_2 = True

# 14j
EBV_14J, RV_14J = -1.37, 1.4
AV_14J, P_14J = 1.85, -2.1

# 12cu
EBV_12CU, RV_12CU = -1.07, 2.59


################################################################################
### LOADERS ####################################################################

def load_14j_colors(phases, filters, zp):
    sn14j  = l.get_14j()
    
    # get 14j photometry at BMAX
    sn14j_colors = {i:{} for i in xrange(len(phases))}
    for f in filters:
        band_phases = np.array([d['phase'] for d in sn14j[f]])
        try:
            band_colors = np.array([(d['Vmag']-d['AV'])-(d['mag']-d['AX']) for d in sn14j[f]])
        except:
            band_colors = np.array([0.0 for d in sn14j[f]])

        sn14j_int = interp1d(band_phases, band_colors)
        for i, phase in enumerate(phases):
            sn14j_colors[i][f] = float(sn14j_int(phase))
        
    return sn14j_colors


def load_12cu_colors(phases, filters, zp):
    prefix = zp['prefix']

    # correct for Milky Way extinction
    sn12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)
    sn12cu = l.interpolate_spectra(phases, sn12cu)

    if type(sn12cu) == type(()):  # convert from tuple to list if just one phase
        sn12cu = [sn12cu]
    
    sn12cu_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')/zp['V']) for t in sn12cu]

    sn12cu_colors = {i:{} for i in xrange(len(phases))}
    for f in filters:
        band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)/zp[f]) for t in sn12cu]
        band_colors = np.array(sn12cu_vmags)-np.array(band_mags)
        for i, color in enumerate(band_colors):
            sn12cu_colors[i][f] = color
        
    return sn12cu_colors


################################################################################
### GENERAL PLOTTING FUNCTION ##################################################

def plotexcess(phases, name, loader, EBV, RV, filters, zp, ax, AV=0.0, P=0.0, plotpl=False):
    
    print "Plotting",name,"..."
    
    ref = loader(phases, filters, zp)
    prefix = zp['prefix']
    filter_eff_waves = [snc.get_bandpass(prefix+f).wave_eff for f in filters]

    # get 11fe synthetic photometry at BMAX, get ref sn color excesses at BMAX
    sn11fe = l.interpolate_spectra(phases, l.get_11fe())

    if type(sn11fe) == type(()):  # convert from tuple to list if just one phase
        sn11fe = [sn11fe]
    
    for i, phase, sn11fe_phase in izip(xrange(len(phases)), phases, sn11fe):
        
        # calculate sn11fe band magnitudes
        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)/zp[f])
                       for f in filters}
        
        # calculate V-X colors for sn11fe
        sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]

        # make list of colors of reference supernova for given phase i
        ref_colors = [ref[i][f] for f in filters]

        # get colors excess of reference supernova compared for sn11fe
        phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)

        # convert effective wavelengths to inverse microns then plot
        eff_waves_inv = (10000./np.array(filter_eff_waves))
        mfc_color = plt.cm.gist_rainbow(abs(phase/24.))        
        plt.plot(eff_waves_inv, phase_excesses, 's', color=mfc_color,
                 ms=8, mec='none', mfc=mfc_color, alpha=0.8)


    x = np.arange(3000,10000,10)
    xinv = 10000./x
    ftz_curve = redden_fm(x, np.zeros(x.shape), EBV, RV, return_excess=True)
    plt.plot(xinv, ftz_curve, 'k--')

    if plotpl:
        # plot PL curve
        gpl_curve = redden_pl(x, np.zeros(x.shape), AV, P, return_excess=True)
        plt.plot(xinv, gpl_curve, 'k-')
    
 
    ax.set_title(name+': Color Excess at B-maximum (with '+prefix[:-1]+' filters)')
    plt.ylabel('$E(V-X)$')
    plt.xlabel('Wavelength ($1 / \mu m$)')
    plt.xlim(1.0, 3.0)


def plot_phase_excesses(phases, name, loader, red_type, EBVS, RVS, filters, zp,
                        plotphot=True, ploterr=True, rederr=0.3):
    
    if red_type == 'fm':
        red_law = redden_fm
        linestyle = '--'
        llabel = 'FTZ'
    elif red_type == 'pl':
        red_law = redden_pl2
        linestyle = '-'
        llabel = 'Power-Law'
        
    print "Plotting excesses of",name,"with",red_type,"RV fit..."
    
    ref = loader(phases, filters, zp)
    prefix = zp['prefix']
    filter_eff_waves = [snc.get_bandpass(prefix+f).wave_eff for f in filters]

    # get 11fe synthetic photometry at BMAX, get ref sn color excesses at BMAX
    sn11fe = l.interpolate_spectra(phases, l.get_11fe())

    if type(sn11fe) == type(()):  # convert from tuple to list if just one phase
        sn11fe = [sn11fe]
    
    numrows = (len(phases)-1)//5 + 1
    
    for i, phase, sn11fe_phase in izip(xrange(len(phases)), phases, sn11fe):
        ax = plt.subplot(numrows, 5, i+1)
        
        # calculate sn11fe band magnitudes
        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)/zp[f])
                       for f in filters}
        
        # calculate V-X colors for sn11fe
        sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]

        # make list of colors of reference supernova for given phase i
        ref_colors = [ref[i][f] for f in filters]

        # get colors excess of reference supernova compared for sn11fe
        phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)

        # convert effective wavelengths to inverse microns then plot
        eff_waves_inv = (10000./np.array(filter_eff_waves))
        mfc_color = plt.cm.gist_rainbow(abs(phase/24.))   
        if plotphot:     
            plt.plot(eff_waves_inv, phase_excesses, 's', color=mfc_color,
                     ms=8, mec='none', mfc=mfc_color, alpha=0.8)
        
        if red_type == 'fm':
            EBV = -EBVS[i]
        elif red_type == 'pl':
            EBV = EBVS[i]
        
        x = np.arange(3000,10000,10)
        xinv = 10000./x
        red_curve = red_law(x, np.zeros(x.shape), EBV, RVS[i], return_excess=True)
        redln, = plt.plot(xinv, red_curve, 'k'+linestyle, label=llabel)
        
        ERROR = rederr
        if ploterr:
            red_curve_upper = red_law(x, np.zeros(x.shape), EBV, RVS[i]-ERROR, return_excess=True)
            red_curve_lower = red_law(x, np.zeros(x.shape), EBV, RVS[i]+ERROR, return_excess=True)
            plt.plot(xinv, red_curve_upper, 'k:')
            plt.plot(xinv, red_curve_lower, 'k:')
            ax.fill_between(xinv, red_curve_lower, red_curve_upper, facecolor='black', alpha=0.1)
        
            plt_text = '$E(B-V)$: $'+str(round(EBVS[i],2))+'\pm'+str(ERROR)+'$'+ \
                       '\n'+'$R_V$: $'+str(round(RVS[i],2))+'$'
        else:
            plt_text = '$E(B-V)$: $'+str(round(EBVS[i],2))+'$' \
                       '\n'+'$R_V$: $'+str(round(RVS[i],2))+'$'
        
        if plotphot:
            ax.text(.95, .95, plt_text, size=FONT_SIZE,
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)
        
        ax.set_title('phase: '+str(phase))
        ax.legend(loc=3, prop={'size':FONT_SIZE})
                     
        if i%5 == 0:
            plt.ylabel('$E(V-X)$')
        if i>=(numrows-1)*5:
            plt.xlabel('Wavelength ($1 / \mu m$)')
        plt.xlim(1.0, 3.0)


################################################################################
### MAIN #######################################################################


def main1():
    fig = plt.figure()
    phases = [-3.5, -1.4, 6.5, 8.5, 11.5, 14.5, 16.5, 18.5, 21.5, 23.5]
    
    ax1 = plt.subplot(1,2,1)
    plotexcess(phases, 'SN2014J', load_14j_colors, EBV_14J, RV_14J,
			   filters_vis, zp_not, ax1, AV_14J, P_14J, plotpl=True)
    ax2 = plt.subplot(1,2,2)
    plotexcess(phases, 'SN2012CU', load_12cu_colors, EBV_12CU, RV_12CU,
               filters_bucket, zp_bucket, ax2)

    # config colorbar
    fig.subplots_adjust(right=0.85)
    cmap = mpl.cm.gist_rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=24)
    cax = fig.add_axes([0.87, 0.15, 0.01, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Days before/after date of B-Maximum')

    # custom legend
    p1, = plt.plot(np.array([]), np.array([]), 's', ms=8, mec='none', mfc='k', alpha=0.3)
    p2, = plt.plot(np.array([]), np.array([]), 'k--')
    p3, = plt.plot(np.array([]), np.array([]), 'k-')
    ax1.legend([p1, p2, p3],
               ['SN2014J Photometry',
                'FTZ: $E(B-V)='+str(EBV_14J)+'$, $R_V='+str(RV_14J)+'$',
                'Power-Law: $A_V='+str(AV_14J)+'$, $P='+str(P_14J)+'$'
                ])
    ax2.legend([p1, p2, p3],
               ['SN2012CU Synthetic Phot.',
                'FTZ: $E(B-V)='+str(EBV_12CU)+'$, $R_V='+str(RV_12CU)+'$'
                ])


def main2(filters, zp):
    from fitexcess import get_12cu_excess_fit
    
    EBVS, RVS, AVS, phases = get_12cu_excess_fit(RED_TYPE, filters, zp)
    
    fig = plt.figure()
    plot_phase_excesses(phases, 'SN2012CU', load_12cu_colors,
                        RED_TYPE, EBVS, RVS, filters, zp,
                        ploterr=PLOT_ERROR_1, rederr=ERROR)
    
    
    other = ('fm','pl')[RED_TYPE == 'fm']
    if PLOT_OTHER:
        EBVS, RVS, AVS, phases = get_12cu_excess_fit(other, filters, zp)
        
        plot_phase_excesses(phases, 'SN2012CU', load_12cu_colors,
                            other, EBVS, RVS, filters, zp,
                            plotphot=False, ploterr=PLOT_ERROR_2, rederr=ERROR)
    
    if RED_TYPE == 'fm':
        appendtxt = ' ($R_V$ fit using FTZ reddening law)'
    elif RED_TYPE == 'pl':
        appendtxt = ' ($R_V$ fit using Power-Law reddening)'
        
    fig.suptitle('SN2012CU: Color Excess Per Phase'+appendtxt,
                 fontsize=18)

    
def plot_ftz_curves():
    fig = plt.figure()
    cmap = mpl.cm.gist_rainbow
    
    RVS = np.arange(2.0, 3.5, .2)
    
    x = np.arange(3000,10000,10)
    xinv = 10000./x
    for RV in RVS:
        c = cmap((max(RVS)-RV)/min(RVS))
        ftz_curve = redden_fm(x, np.zeros(x.shape), -1.07, RV, return_excess=True)
        plt.plot(xinv, ftz_curve, '-', color=c)
        
    plt.ylabel('$E(V-X)$')
    plt.xlabel('Wavelength ($1 / \mu m$)')
    plt.title('FTZ Color Excess Curves')
    
    # config colorbar
    fig.subplots_adjust(right=0.85)
    norm = mpl.colors.Normalize(vmin=min(RVS), vmax=max(RVS))
    cax = fig.add_axes([0.87, 0.15, 0.01, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('$R_V$')
    
    
    
if __name__=='__main__':
    
    filters_vis = 'UBVRI'
    zp_top = l.load_filters('tophat_')
    zp_not = l.load_filters('NOT_')
    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
    
    if not PLOT_PHASES:
        main1()
    else:
        main2(filters_bucket, zp_bucket)
    #plot_ftz_curves()
    
    '''
    do power law fit and plot.
    '''
    
    plt.show()
    
