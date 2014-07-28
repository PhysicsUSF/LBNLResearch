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
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sncosmo as snc

from pprint import pprint



def load_14j(phase, filters, zp):
    sn14j  = l.get_14j()
    
    # get 14j photometry at BMAX
    sn14j_colors = {}
    for f in filters:
        band_phases = np.array([d['phase'] for d in sn14j[f]])
        try:
            band_colors = np.array([(d['Vmag']-d['AV'])-(d['mag']-d['AX']) for d in sn14j[f]])
        except:
            band_colors = np.array([0.0 for d in sn14j[f]])
        
        sn14j_colors[f] = (interp1d(band_phases, band_colors))(phase)
        
    return sn14j_colors


def load_12cu(phase, filters, zp):
    prefix = zp['prefix']
    sn12cu = l.get_12cu()
    
    # get 12cu photometry at BMAX
    SN12CU_MW = dict( zip( 'UBVRI', [0.117, 0.098, 0.074, 0.058, 0.041] ))
    
    sn12cu = l.interpolate_spectra(phase, sn12cu)
    sn12cu_vmag = -2.5*np.log10(sn12cu[1].bandflux(prefix+'V')/zp['V'])-SN12CU_MW['V']

    sn12cu_colors = {}
    for f in filters:
        band_mag = -2.5*np.log10(sn12cu[1].bandflux(prefix+f)/zp[f])-SN12CU_MW[f]
        sn12cu_colors[f] = sn12cu_vmag-band_mag
        
    return sn12cu_colors


def plotexcess(phase, name, loader, EBV, RV, filters, zp, ax, plotfm=True, plotpl=True):
    ref_bmax_colors = loader(phase, filters, zp)
    
    prefix = zp['prefix']
    filter_eff_waves = [snc.get_bandpass(prefix+f).wave_eff for f in filters]

    # get 11fe synthetic photometry at BMAX, get ref sn color excesses at BMAX
    sn11fe_bmax = l.interpolate_spectra(phase, l.get_11fe())
    #sn11fe_bmax_red =l.interpolate_spectra(phase, l.get_11fe('fm', EBV, RV))
    
    sn11fe_bmax_mags = {f:-2.5*np.log10(sn11fe_bmax[1].bandflux(prefix+f)/zp[f]) for f in filters}
    #sn11fe_bmax_mags_red = {f:-2.5*np.log10(sn11fe_bmax_red[1].bandflux(prefix+f)/zp[f]) for f in filters}
    
    sn11fe_bmax_colors = [sn11fe_bmax_mags['V']-sn11fe_bmax_mags[f] for f in filters]
    #sn11fe_bmax_colors_red = [sn11fe_bmax_mags_red['V']-sn11fe_bmax_mags_red[f] for f in filters]

    ref_bmax_colors = [ref_bmax_colors[f] for f in filters]
    bmax_excesses = np.array(ref_bmax_colors)-np.array(sn11fe_bmax_colors)
    
    #sn11fe_bmax_excesses = np.array(sn11fe_bmax_colors_red)-np.array(sn11fe_bmax_colors)

    x = np.arange(3000,10000,10)
    ftz_curve = redden_fm(x, np.zeros(x.shape), EBV, RV, return_excess=True)
    
    AV, P = 1.85, -2.1
    gpl_curve = redden_pl(x, np.zeros(x.shape),  AV,  P, return_excess=True)

    # convert Angstroms to inverse microns
    xinv = 10000./x
    eff_waves_inv = (10000./np.array(filter_eff_waves)) #+ 0.12
    
    plt.plot(eff_waves_inv, bmax_excesses, 'go', ms=7, alpha=0.8, label=name+' photometry')
    if plotfm:
        plt.plot(xinv, ftz_curve, 'k--', label='FTZ: $E(B-V)='+str(EBV)+'$, $R_V='+str(RV)+'$')
    if plotpl:
        plt.plot(xinv, gpl_curve, 'k-',  label='Power-Law: $A_V='+str(AV)+'$, $P='+str(P)+'$')

    # annotate filters


##    for i, f in enumerate(filters):
##        plt.text(eff_waves_inv[i]-.1, bmax_excesses[i]+.1, f) 
    
    ax.set_title(name+': Color Excess at B-maximum (with '+prefix+' filters)')
    plt.ylabel('$E(V-X)$')
    plt.xlabel('Wavelength ($1 / \mu m$)')
##    plt.legend()



def main():
    
    filters = 'UBVRI'
    zp_top = l.load_filters('tophat_')
    zp_not = l.load_filters('NOT_')

    # Best fits for;    EBV     RV
    #       2014J*...   -1.37   1.4    *from Amanullah (2014)
    #       2012CU...   -1.07   2.59

    plt.figure()
    for phase in np.arange(-3., 25., 1.5):
        print phase
        
        ax1 = plt.subplot(1,2,1)
        plotexcess(phase, 'SN2014J', load_14j,  -1.37,  1.4, filters, zp_not, ax1)
        
        ax2 = plt.subplot(1,2,2)
        plotexcess(phase, 'SN2012CU', load_12cu, -1.07, 2.59, filters, zp_not, ax2, plotpl=False)
    
    plt.show()
    
'''
RESULTS:

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

if __name__=='__main__':
    main()
