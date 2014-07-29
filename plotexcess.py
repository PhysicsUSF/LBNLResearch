'''
::Author::
Andrew Stocker

::Description::
This program will plot the color excess plot for 14J and 12CU with various reddening laws.

::Last Modified::
07/24/2014

'''
import loader as l
from loader import redden_fm, redden_pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sncosmo as snc
from itertools import izip

from pprint import pprint

## vars ##

# 14j
EBV_14J, RV_14J = -1.37, 1.4
AV_14J, P_14J = 1.85, -2.1

# 12cu
EBV_12CU, RV_12CU = -1.07, 2.59
AV_12CU, P_12CU = RV_12CU*(-EBV_12CU), np.log((1/RV_12CU)+1)/np.log(0.8)


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
    sn12cu = l.get_12cu()
    
    # get 12cu photometry at BMAX
    #SN12CU_MW = dict( zip( 'UBVRI', [0.117, 0.098, 0.074, 0.058, 0.041] ))
    
    sn12cu = l.interpolate_spectra(phases, sn12cu)

    sn12cu_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')/zp['V']) for t in sn12cu]  # -SN12CU_MW['V']

    sn12cu_colors = {i:{} for i in xrange(len(phases))}
    for f in filters:
        band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)/zp[f]) for t in sn12cu]  # -SN12CU_MW[f]
        band_colors = np.array(sn12cu_vmags)-np.array(band_mags)

        for i, color in enumerate(band_colors):
            sn12cu_colors[i][f] = color
        
    return sn12cu_colors


################################################################################
### TOPHAT FILTER GENERATOR ####################################################

def generate_buckets(low, high, n):
    '''
    function to generate [n] tophat filters within the
    range of [low, high], with one 'V' filter centered at
    the BESSEL-V filter's effective wavelength (5417.2 Angstroms).
    '''
    V_EFF = 5417.2
    STEP_SIZE = 2
    ZP_CACHE = {'prefix':'bucket_'}

    hi_cut, lo_cut = high-V_EFF, V_EFF-low
    a = (n-1)/(1+(hi_cut/lo_cut))
    A, B = np.round(a), np.round(n-1-a)

    lo_bw, hi_bw = lo_cut/(A+0.5), hi_cut/(B+0.5)
    lo_diff = lo_cut-(A+0.5)*hi_bw
    hi_diff = hi_cut-(B+0.5)*lo_bw

    idx = np.argmax((lo_diff, hi_diff))
    BW = (hi_bw, lo_bw)[idx]

    LOW = (low+lo_diff, low)[idx]

    for i in xrange(n):
        start = LOW+i*BW
        end = LOW+(i+1)*BW
        
        wave = np.arange(start, end, STEP_SIZE)
        trans = np.ones(wave.shape[0])
        trans[0]=trans[-1]=0

        if not abs(5417.2-(start+end)/2) < 1:
            filter_name = 'bucket_'+str(i)
        else:
            filter_name = 'bucket_V'
            V_IDX = i
            
        try:
            snc.get_bandpass(filter_name)
        except:
            bandpass = snc.Bandpass(wave, trans)
            snc.registry.register(bandpass, filter_name)

        zpsys = snc.get_magsystem('vega')
        zp_phot = zpsys.zpbandflux(filter_name)
        
        if filter_name == 'bucket_V':
            ZP_CACHE['V'] = zp_phot
        else:
            ZP_CACHE[str(i)] = zp_phot

    FILTERS = [str(i) for i in xrange(n)]
    FILTERS[V_IDX] = 'V'
    
    return FILTERS, ZP_CACHE


################################################################################
### GENERAL PLOTTING FUNCTION ##################################################

def plotexcess(phases, name, loader, EBV, RV, AV, P, filters, zp, ax, plotpl=True):
    print "Plotting",name,"..."
    ref = loader(phases, filters, zp)
    
    prefix = zp['prefix']
    filter_eff_waves = [snc.get_bandpass(prefix+f).wave_eff for f in filters]

    # get 11fe synthetic photometry at BMAX, get ref sn color excesses at BMAX
    sn11fe = l.interpolate_spectra(phases, l.get_11fe())

    
    for i, phase, sn11fe_phase in izip(xrange(len(phases)), phases, sn11fe):
        print phase,"..."

        sn11fe_mags = {f : -2.5*np.log10(sn11fe_phase[1].bandflux(prefix+f)/zp[f])
                       for f in filters}
        
        sn11fe_colors = [sn11fe_mags['V']-sn11fe_mags[f] for f in filters]

        ref_colors = [ref[i][f] for f in filters]
        
        phase_excesses = np.array(ref_colors)-np.array(sn11fe_colors)

        eff_waves_inv = (10000./np.array(filter_eff_waves)) #+ 0.12
        mfc_color = plt.cm.gist_rainbow(abs(phase/np.max(phases)))
        
        plt.plot(eff_waves_inv, phase_excesses, 's', color=mfc_color,
                 ms=8, mec='none', mfc=mfc_color, alpha=0.8)

    # plot FTZ curve
    x = np.arange(3000,10000,10)
    ftz_curve = redden_fm(x, np.zeros(x.shape), EBV, RV, return_excess=True)
    xinv = 10000./x
    plt.plot(xinv, ftz_curve, 'k--')

    if plotpl:
        # plot PL curve
        gpl_curve = redden_pl(x, np.zeros(x.shape), AV, P, return_excess=True)
        plt.plot(xinv, gpl_curve, 'k-')
    
 
    ax.set_title(name+': Color Excess at B-maximum (with '+prefix[:-1]+' filters)')
    plt.ylabel('$E(V-X)$')
    plt.xlabel('Wavelength ($1 / \mu m$)')
    plt.xlim(1.0, 3.0)


################################################################################
### MAIN #######################################################################

if __name__=='__main__':
    
    filters_vis = 'UBVRI'
    zp_top = l.load_filters('tophat_')
    zp_not = l.load_filters('NOT_')

    filters_bucket, zp_bucket = generate_buckets(3300, 9700, 20)

    fig = plt.figure()
    phases = np.arange(-3., 25., 1.5)
    
    ax1 = plt.subplot(1,2,1)
    plotexcess(phases, 'SN2014J', load_14j_colors, EBV_14J, RV_14J, AV_14J, P_14J, filters_vis, zp_not, ax1)
    ax2 = plt.subplot(1,2,2)
    plotexcess(phases, 'SN2012CU', load_12cu_colors, EBV_12CU, RV_12CU, AV_12CU, P_12CU,
               filters_bucket, zp_bucket, ax2, plotpl=False)

    # config colorbar
    fig.subplots_adjust(right=0.85)
    cmap = mpl.cm.gist_rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=24)
    cax = fig.add_axes([0.87, 0.15, 0.01, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Days before/after date of B-Maximum')

    # custom legend
    p1, = plt.plot(np.array([]), np.array([]), 's', ms=8, mec='none', mfc='r', alpha=0.8)
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
    plt.show()
    
    
'''
RESULTS AT BMAX:

from plotcolors.py...

        U Plotting...
                SN2012CU @BMAX
                REDDENED V-X: [ 12.54478879] [(0.0, -1.4453838324745139)]
                PRISTINE V-X: [ 10.02714581] [(0.0, 0.44521803614300914)]
                E(V-X): -1.89060186862
        B Plotting...
                SN2012CU @BMAX
                REDDENED V-X: [ 12.54478879] [(0.0, -1.0838071436716756)]
                PRISTINE V-X: [ 10.02714581] [(0.0, -0.012092136584524127)]
                E(V-X): -1.07171500709
        R Plotting...
                SN2012CU @BMAX
                REDDENED V-X: [ 12.54478879] [(0.0, 0.56485297025714409)]
                PRISTINE V-X: [ 10.02714581] [(0.0, -0.07808337311118585)]
                E(V-X): 0.642936343368
        I Plotting...
                SN2012CU @BMAX
                REDDENED V-X: [ 12.54478879] [(0.0, 1.0220264270188153)]
                PRISTINE V-X: [ 10.02714581] [(0.0, -0.15394727646707906)]
                E(V-X): 1.17597370349

from this file:
with tophat filters...

	REDDENED SN11FE E(V-X)
	U -1.89589158316
	B -1.07468328753
	V 0.0
	R 0.644745541401
	I 1.17933364238

with NOT filters...

	REDDENED SN11FE E(V-X)
	U -1.75870462292
	B -1.05423954507
	V 0.0
	R 0.640207528848
	I 1.30813200355
'''

