'''
::Author::
Andrew Stocker

::Description::
This program will fit RV based on color excess of 2012cu

::Last Modified::
08/01/2014

'''
import loader as l
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from itertools import izip
from loader import redden_fm, redden_pl
from pprint import pprint
from scipy.interpolate import interp1d
from scipy.optimize import leastsq as lsq


## vars ##
N_BUCKETS = 20

# 14j
EBV_14J, RV_14J = -1.37, 1.4
AV_14J, P_14J = 1.85, -2.1

# 12cu
EBV_12CU, RV_12CU = -1.07, 2.59
AV_12CU, P_12CU = RV_12CU*(-EBV_12CU), np.log((1/RV_12CU)+1)/np.log(0.8)


################################################################################
### LEAST-SQ FITTER ############################################################

def lsq_excess_fit(ref_excess_dict, EBV, rv_guess, filters, zp):
    prefix = zp['prefix']
    filter_eff_waves = np.array([snc.get_bandpass(prefix+f).wave_eff for f in filters])
    ref_excess = np.array([ref_excess_dict[f] for f in filters])
    
    def lsq_func(Y):
        RV = Y[0]
        ftz_curve = redden_fm(filter_eff_waves,
                              np.zeros(filter_eff_waves.shape),
                              -EBV, RV,
                              return_excess=True)
        
        return ftz_curve-ref_excess
    
    Y = np.array([rv_guess])
    valid_phases = {}
    return lsq(lsq_func, Y)


################################################################################
### LOADERS ####################################################################

def get_ebvs(sn11fe, sn12cu):
    if not np.array_equal([t[0] for t in sn11fe], [t[0] for t in sn12cu]):
        raise ValueError
    # requires not filters to be imported
    zp = l.load_filters('NOT_')
    prefix = zp['prefix']
    ebvs = {}
    for i in xrange(len(sn11fe)):
        s1, s2 = sn11fe[i][1], sn12cu[i][1]  
        s1b, s1v = s1.bandflux(prefix+'B'), s1.bandflux(prefix+'V')
        s2b, s2v = s2.bandflux(prefix+'B'), s2.bandflux(prefix+'V')
        s1bmag = -2.5*np.log10( s1b/zp['B'] )
        s1vmag = -2.5*np.log10( s1v/zp['V'] )
        s2bmag = -2.5*np.log10( s2b/zp['B'] )
        s2vmag = -2.5*np.log10( s2v/zp['V'] )
        ebvs[i] = (s2bmag-s1bmag) - (s2vmag-s1vmag)
    return ebvs


def load_12cu_excess(sn12cu, sn11fe, filters, zp):
    prefix = zp['prefix']

    ebvs = []

    sn12cu = filter(lambda t: -3.6<t[0]<25, sn12cu)
    phases = [t[0] for t in sn12cu]
    sn11fe = l.interpolate_spectra(phases, sn11fe)

    sn11fe_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')/zp['V']) for t in sn11fe]
    sn12cu_vmags = [-2.5*np.log10(t[1].bandflux(prefix+'V')/zp['V']) for t in sn12cu]

    sn12cu_excess = {i:{} for i in xrange(len(phases))}
    for f in filters:
        sn11fe_band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)/zp[f]) for t in sn11fe]
        sn11fe_band_colors = np.array(sn11fe_vmags)-np.array(sn11fe_band_mags)
        sn12cu_band_mags = [-2.5*np.log10(t[1].bandflux(prefix+f)/zp[f]) for t in sn12cu]
        sn12cu_band_colors = np.array(sn12cu_vmags)-np.array(sn12cu_band_mags)
        for i in xrange(len(phases)):
            sn12cu_excess[i][f] = sn12cu_band_colors[i] - sn11fe_band_colors[i]

    return sn12cu_excess, get_ebvs(sn11fe, sn12cu), phases


################################################################################
### MAIN #######################################################################

def get_12cu_ftz_excess_fit(filters, zp):
    
    #sn11fe = l.get_11fe(loadptf=False, loadmast=False)
    sn11fe = l.get_11fe()

    ### 12CU ###
    sn12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)  # correct for Milky Way extinction

    sn12cu_excess, sn12cu_ebvs, phases = load_12cu_excess(sn12cu, sn11fe, filters, zp)

    EBVS = []
    RVS = []
    AVS = []
    
    print "Doing Least-Sq Fit on SN2012CU with "+zp['prefix'][:-1]+" Filter Set..."
    for i, phase in enumerate(phases):
        ebv = sn12cu_ebvs[i]
        lsq_out = lsq_excess_fit(sn12cu_excess[i], ebv, 2.6, filters, zp)
        rv = lsq_out[0][0]
        av = ebv*rv
        
        if __name__=='__main__':
            print "phase:",phase,", ebv:",ebv,"..."
            print "\tRESULT:"
            print "\tRV:",rv,"\tAV:",av
            print

        EBVS.append(ebv)
        RVS.append(rv)
        AVS.append(av)
        
    return EBVS, RVS, AVS, phases


################################################################################


if __name__=='__main__':
    filters_bucket, zp_bucket = l.generate_buckets(3300, 9700, N_BUCKETS, inverse_microns=True)
    
    EBVS, RVS, AVS, phases = get_12cu_ftz_excess_fit(filters_bucket, zp_bucket)
    
    plt.figure()
    plt.plot(phases, RVS, 'rs-', mfc='none', ms=7, mew=2, mec='r', label='$R_V$')
    plt.plot(phases, AVS, 'gs-', mfc='none', ms=7, mew=2, mec='g', label='$A_V$')
    plt.plot(phases, EBVS, 'bs-', mfc='none', ms=7, mew=2, mec='b', label='$E(B-V)$')
    plt.title('SN2012CU Extinction by Phase')
    plt.legend(loc=7)

    ############
    plt.show()

    



        

