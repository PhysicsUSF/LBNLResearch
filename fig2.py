import loader as l
import numpy as np
import matplotlib.pyplot as plt
import sncosmo as snc

from pprint import pprint
from itertools import izip
from copy import deepcopy


################################################################################


def find_valid(array,value_array,n):
    array = np.array(array)
    return_array = np.zeros(len(value_array), dtype=np.bool)
    for i, x in enumerate(value_array):
        idx = (np.abs(array-x)).argmin()
        return_array[i] = abs(array[idx]-x) <= n
        
    return return_array


def calc_v_band_mags(spectra, zp):
    # spectra is a list of tuples where the first index is the phase
    #  and the second index in an snc.Spectrum() object
    
    vflux = np.zeros(len(spectra))
    for i, t in enumerate(spectra):
        try:
            vflux[i] = t[1].bandflux('tophat_V')/ zp['V']
        except:
            vflux[i] = np.inf
    
    vmags = -2.5*np.log10( vflux/zp['V'] )
    phases = [s[0] for s in spectra]
    return phases, vmags


def calc_colors(f, spectra, vmags, zp):
    # spectra is a list of tuples where the first index is the phase
    #  and the second index in an snc.Spectrum() object
    filter_name = 'tophat_' + f

    bflux = np.zeros(len(spectra))
    for i, t in enumerate(spectra):
        try:
            bflux[i] = t[1].bandflux(filter_name) / zp[f]
        except:
            bflux[i] = np.inf
    
    bmags = -2.5*np.log10( bflux/zp[f] )
    
    phases = [s[0] for s in spectra]
    colors = vmags-bmags
    
    return filter(lambda tup: not np.isinf(tup[1]), izip(phases, colors))


def lsq_color_fit(sn14j, EBV, rv_guess, zp):
    N_DAYS = 1.0
    
    from scipy.optimize import leastsq as lsq

    filters = 'UBRI'

    def lsq_func(Y):
        print Y
        
        RV = Y[0]
        sn11fe = l.get_11fe('fm', ebv=EBV, rv=RV)

        sn11fe_colors = np.array([])
        sn14j_colors = np.array([])
        for f in filters:
            FDATA = sn14j[f]

            phases = [d['phase'] for d in FDATA]
    
            valid = find_valid([t[0] for t in sn11fe], phases, N_DAYS)
            valid_phases = np.array(phases)[valid]

            sn14j_bandcolors = [(d['Vmag']-d['AV'])-(d['mag']-d['AX']) for d in FDATA if d['phase'] in valid_phases]
            
            sn11fe_int = l.interpolate_spectra(valid_phases, sn11fe)
            temp, sn11fe_vmags = calc_v_band_mags(sn11fe_int, zp)

            sn11fe_bandcolors = calc_colors(f, sn11fe_int, sn11fe_vmags, zp)
            sn11fe_bandcolors = [t[1] for t in sn11fe_bandcolors]

            sn14j_colors = np.concatenate(( sn14j_colors, sn14j_bandcolors ))
            sn11fe_colors = np.concatenate(( sn11fe_colors, sn11fe_bandcolors ))

        return sn14j_colors - sn11fe_colors
    
    Y = np.array([rv_guess])

    return lsq(lsq_func, Y)


################################################################################


def main():
    FILTERS = 'UBRI'

    zp     = l.load_filters()
    sn14j  = l.get_14j()

    # vars
    EBV, R_V = 1.29,  1.4
    A_V,   P =  1.85, -2.1

    # best ebv  = 1.29 ?  (from Amanullah)
    lsq_out = lsq_color_fit(sn14j, EBV, R_V, zp)

    print lsq_out

    BEST_RV = lsq_out[0][0]

    

    ################################################################################
    
    row_ylims = {  'U' : [-5,  0],
                   'B' : [-2.5,1.5],
                   'V' : [-2.5,1.5],
                   'R' : [-2.5,1.5],
                   'I' : [0.2, 2.0]
                }

    
    SN2011FE_FMR = l.get_11fe('fm', ebv=EBV, rv=BEST_RV)
    SN2011FE_PLR = l.get_11fe('pl', av=A_V, p=P)

    phases, VMAGS_SN2011FE_FMR = calc_v_band_mags(SN2011FE_FMR, zp)
    phases, VMAGS_SN2011FE_PLR = calc_v_band_mags(SN2011FE_PLR, zp)

    fig = plt.figure()
    index = 1
    
    for FILTER in FILTERS:
        ax = plt.subplot(2,2,index)

        # 2014J data
        FDATA = sn14j[FILTER]
        sn14j_phases = [epoch['phase'] for epoch in FDATA]
        sn14j_mags = [(epoch['Vmag']-epoch['AV'])-(epoch['mag']-epoch['AX']) for epoch in FDATA]
        
        p3, = plt.plot(sn14j_phases, sn14j_mags, 'bo')

        # reddened 2011FE data
        print FILTER + " Plotting..."

        filtered_fmr = calc_colors(FILTER, SN2011FE_FMR, VMAGS_SN2011FE_FMR, zp)
        filtered_plr = calc_colors(FILTER, SN2011FE_PLR, VMAGS_SN2011FE_PLR, zp)

        p1, = plt.plot([t[0] for t in filtered_fmr], [t[1] for t in filtered_fmr], 'r--')
        p2, = plt.plot([t[0] for t in filtered_plr], [t[1] for t in filtered_plr], 'g--')

        
        # format subplot
        if index%3 == 1:
            plt.ylabel("$V-X$")
        if index<3:
            ax.xaxis.set_ticklabels([])
        else:
            plt.xlabel("Phase (relative B-max)")
        plt.ylim(row_ylims[FILTER])
        plt.xlim(-5,35)
        ax.set_title(FILTER)
        index += 1

    # format figure
    fig.suptitle("Broadband Colors vs. Phase", fontsize=18)
    fig.legend([p1, p2, p3],
               ["FTZ: $E(B-V) = "+str(round(EBV,2))+"$, $R_V = "+str(round(BEST_RV,2))+"$",
                "PowerLaw: $A_V = 1.77$, $p=-2.1$",
                "Amanaullah 2014J"],
               loc=3,
               ncol=3,
               mode="expand"
               )
    plt.show()


if __name__ == "__main__":
    main()
