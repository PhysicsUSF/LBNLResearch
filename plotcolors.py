'''
::Author::
Andrew Stocker

::Descript::
This program will plot the photometry for 14J and 12CU in the UBRI bands, and will also
do a best fit with an artificially reddened 11FE in order to determine the RV for 14J
and 12CU respectively.

::Last Modified::
07/22/2014

'''
import loader as l
import numpy as np
import matplotlib.pyplot as plt
import sncosmo as snc

from pprint import pprint
from itertools import izip
from copy import deepcopy


# matplotlib vars
LO, HI = -5, 30  # cutoffs for phases to show
SN11FE_PLOT_ALPHA = 0.8
MARKER_SIZE = 7
CROSS_SIZE = 2

# valid day range for interpolation, for use in 11fe reddening fits
N_DAYS = 2.0

# filters to use in fit (must be subset of UBVRI)
FILTERS = 'UBRI'

# vars for reddening 11fe to match 14j, best ebv  = 1.29 ?  (from Amanullah)
EBV_14J, RV_14J = -1.29,  1.3

# vars for reddening 11fe to match 12cu, best ebv calculated from 12cu synthetic phot.
EBV_12CU, RV_12CU = -1.067, 2.43


################################################################################
### FUNCTIONS ##################################################################


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
            vflux[i] = t[1].bandflux('tophat_V')
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
            bflux[i] = t[1].bandflux(filter_name)
        except:
            bflux[i] = np.inf
    
    bmags = -2.5*np.log10( bflux/zp[f] )
    phases = [s[0] for s in spectra]
    colors = vmags-bmags

    out = filter(lambda tup: (not np.isinf(tup[1])) and (not np.isnan(tup[1])),
                 izip(phases, colors))

    return out


def filter_lc(spectra):
    return filter(lambda t: LO<t[0]<HI, spectra)


def lsq_11fe_color_fit(SN, EBV, rv_guess, filters, zp, N_DAYS):
    '''
    This is a helper function to fit an RV to a certain supernova (SN), using
    an artificially reddened 11fe as a template.  EBV is fixed and then 'filters'
    are the filters you want to include in the fit (with 'zp' as zero-points
    dictionary).  The fitter will only use phases from the comparison supernova
    which are no more than N_DAYS from the nearest 11fe phase.  SN needs to be
    in the form (populated by the phases and corresponding V-X colors):

        SN = {'B':[(-6.5, -1.0601018152416941),
                   (-3.5, -1.0864032119751066),
                   (-1.4, -1.08253418576437),
                   (6.5, -1.2348233700462039),
                   (8.5, -1.2831185823642457),
                   (11.5, -1.3001372408790797),
                   ...

        lsq_out, valid_phases = lsq_11fe_color_fit(SN, -1.29, 1.3, 'UBRI', ZP_CACHE, 2.0)

    This function outputs the output of SciPy's least-square fit, and also a
    dictionary with the phases used in the fit, respective to each band:
    
        lsq_out =  (array([ 1.3776701]), 1)
        
        valid_phases =  {'B': array([ -3.5,  -0.7,   1.1,   5.2,   6.9,   7.9,
                                       8.7,  16.7,  20.3,  28.1]),
                         'I': array([ -3.5,  -2.5,  -0.7,   1.1,   3.4,   5.2,
                                       6.9,   7.9,   8.7,  16.7,  20.3,  28.1]),
                         'R': array([ -3.5,  -2.5,  -0.7,   1.1,   3.4,   5.2,
                                       6.9,   7.9,   8.7,  16.7,  20.3,  28.1]),
                         'U': array([ -3.5,  -2.5,   3.4,   5.2,   6.9,   7.9,
                                       8.7,  16.7,  20.3,  28.1])}
    
    '''
    from scipy.optimize import leastsq as lsq

    def lsq_func(Y):
        print Y
        
        RV = Y[0]
        sn11fe = l.get_11fe('fm', ebv=EBV, rv=RV)
                
        sn11fe_colors = np.array([])
        sncomp_colors = np.array([])
        for f in filters:

            # get phases from comparison supernova
            band_data = SN[f]
            phases = [t[0] for t in band_data]

            # interpolate 11fe at 12cu phases which are no more than N_DAYS
            # away from the nearest 11fe phase
            valid = find_valid([t[0] for t in sn11fe], phases, N_DAYS)
            valid_phases[f] = np.array(phases)[valid]
            sn11fe_int = l.interpolate_spectra(valid_phases[f], sn11fe)

            # calculate band magnitudes and then colors for interpolated 11fe
            temp, sn11fe_vmags = calc_v_band_mags(sn11fe_int, zp)
            sn11fe_bandcolors = calc_colors(f, sn11fe_int, sn11fe_vmags, zp)
            sn11fe_phases     = [t[0] for t in sn11fe_bandcolors]
            sn11fe_bandcolors = [t[1] for t in sn11fe_bandcolors]
            sn11fe_colors = np.concatenate(( sn11fe_colors, sn11fe_bandcolors ))

            # get comparison supernova's color at valid phases
            sncomp_bandcolors = [t[1] for t in band_data if t[0] in sn11fe_phases]
            sncomp_colors = np.concatenate(( sncomp_colors, sncomp_bandcolors ))

        
        return sncomp_colors - sn11fe_colors
    
    Y = np.array([rv_guess])
    valid_phases = {}

    return lsq(lsq_func, Y), valid_phases


################################################################################


def main():
    

    zp     = l.load_filters()
    sn14j  = l.get_14j()
    sn12cu = l.get_12cu()

    row_ylims = {  'U' : [-5,  0],
                   'B' : [-2.5,1.5],
                   'R' : [-2.5,1.5],
                   'I' : [0.2, 2.0]
                }

    
    sn11fe_plot_proxy, = plt.plot(np.array([]), np.array([]),
                                  'ro--', mfc='r', mec='r', ms=7, alpha=0.8)

    ############################################################################
    ### SN2014J COMPARISON #####################################################
    print "SN2014J: LEAST SQUARED FIT FOR R_V"
    
    sn14j_dict = {}
    for f in FILTERS:
        FDATA = sn14j[f]
        sn14j_phases = [d['phase'] for d in FDATA]
        sn14j_bandcolors = [(d['Vmag']-d['AV'])-(d['mag']-d['AX']) for d in FDATA]
        sn14j_dict[f] = zip(sn14j_phases, sn14j_bandcolors)
        
    lsq_out, valid_phases = lsq_11fe_color_fit(sn14j_dict, EBV_14J, RV_14J, FILTERS, zp, N_DAYS)
    BEST_RV = lsq_out[0][0]
    sn2011fe_red = l.get_11fe('fm', EBV_14J, rv=1.3)

    print "LSQ OUT:", lsq_out

    fig1 = plt.figure(1)
    index = 1
    for FILTER in FILTERS:
        ax = plt.subplot(2,2,index)
        
        print FILTER + " Plotting..."

        # 2014J data
        band_data = filter_lc( sn14j_dict[FILTER] )
        sn14j_phases = np.array([t[0] for t in band_data])
        sn14j_colors = np.array([t[1] for t in band_data])
        p2, = plt.plot(sn14j_phases, sn14j_colors, 'bo', ms=MARKER_SIZE)

        # reddened 2011FE data
        sn2011fe_int = l.interpolate_spectra(sn14j_phases, sn2011fe_red)
        temp, sn2011fe_vmags = calc_v_band_mags(sn2011fe_int, zp)
        filtered_fmr = filter_lc( calc_colors(FILTER, sn2011fe_int, sn2011fe_vmags, zp) )
        sn11fe_phases = np.array([t[0] for t in filtered_fmr])
        sn11fe_colors = np.array([t[1] for t in filtered_fmr])
        plt.plot(sn11fe_phases, sn11fe_colors, 'ro--', mfc='none', mec='r', ms=MARKER_SIZE)

        # plot points that were used in interpolation
        valid = valid_phases[FILTER]
        sn11fe_valid = filter_lc( filter(lambda t: t[0] in valid, filtered_fmr) )
        sn11fe_valid_phases = [t[0] for t in sn11fe_valid]
        sn11fe_valid_colors = [t[1] for t in sn11fe_valid]
        plt.plot(sn11fe_valid_phases,
                 sn11fe_valid_colors,
                 'ro', mfc='r', mec='r',
                 ms=MARKER_SIZE, alpha=SN11FE_PLOT_ALPHA)

        # plot original data points for 11fe
        temp, sn2011fe_orig_vmags = calc_v_band_mags(sn2011fe_red, zp)
        sn2011fe_orig = filter_lc( calc_colors(FILTER, sn2011fe_red, sn2011fe_orig_vmags, zp) )
        sn2011fe_orig_phases = [t[0] for t in sn2011fe_orig]
        sn2011fe_orig_colors = [t[1] for t in sn2011fe_orig]
        p1, = plt.plot(sn2011fe_orig_phases, sn2011fe_orig_colors, 'g+', mew=CROSS_SIZE)

        # format subplot
        if index%2 == 1:
            plt.ylabel("$V-X$")
        if index>2:
            plt.xlabel("Phase (relative B-max)")
        plt.ylim(row_ylims[FILTER])
        plt.xlim(LO,HI)
        ax.set_title(FILTER)
        index += 1

    # format figure
    fig1.suptitle("2014J: Broadband Colors vs. Phase", fontsize=18)
    fig1.legend([sn11fe_plot_proxy, p1, p2],
                ["FTZ reddened 11fe: $E(B-V) = "+str(round(EBV_14J,2))+"$, $R_V = "+str(round(BEST_RV,2))+"$",
                 "Original 11fe data",
                 "Amanaullah 2014j"],
                loc=3,
                ncol=3,
                mode="expand"
                )


    ############################################################################
    ### SN2012CU COMPARISON ####################################################
    print "SN2012CU: LEAST SQUARED FIT FOR R_V"

    temp, sn12cu_vmags = calc_v_band_mags(sn12cu, zp)
    sn12cu_dict = {}
    
    for f in FILTERS:
        sn12cu_bandcolors = calc_colors(f, sn12cu, sn12cu_vmags, zp)
        sn12cu_phases     = [t[0] for t in sn12cu_bandcolors]
        sn12cu_bandcolors = [t[1] for t in sn12cu_bandcolors]
        sn12cu_dict[f] = zip(sn12cu_phases, sn12cu_bandcolors)
    
    lsq_out, valid_phases = lsq_11fe_color_fit(sn12cu_dict, EBV_12CU, RV_12CU,
                                               FILTERS, zp, N_DAYS)
    BEST_RV = lsq_out[0][0]
    sn2011fe_red = l.get_11fe('fm', EBV_12CU, rv=BEST_RV)

    print "LSQ OUT:", lsq_out

    fig2 = plt.figure(2)
    index = 1
    for FILTER in FILTERS:
        ax = plt.subplot(2,2,index)
        
        print FILTER + " Plotting..."

        # 2012CU data
        band_data = filter_lc( sn12cu_dict[FILTER] )
        sn12cu_phases = np.array([t[0] for t in band_data])
        sn12cu_colors = np.array([t[1] for t in band_data])
        p2, = plt.plot(sn12cu_phases, sn12cu_colors, 'bo', ms=MARKER_SIZE)

        # reddened 2011FE data
        sn2011fe_int = l.interpolate_spectra(sn12cu_phases, sn2011fe_red)
        temp, sn2011fe_vmags = calc_v_band_mags(sn2011fe_int, zp)
        filtered_fmr = filter_lc( calc_colors(FILTER, sn2011fe_int, sn2011fe_vmags, zp) )
        sn11fe_phases = np.array([t[0] for t in filtered_fmr])
        sn11fe_colors = np.array([t[1] for t in filtered_fmr])
        plt.plot(sn11fe_phases, sn11fe_colors, 'ro--', mfc='none', mec='r', ms=MARKER_SIZE)

        # plot points that were used in interpolation
        valid = valid_phases[FILTER]
        sn11fe_valid = filter_lc( filter(lambda t: t[0] in valid, filtered_fmr) )
        sn11fe_valid_phases = [t[0] for t in sn11fe_valid]
        sn11fe_valid_colors = [t[1] for t in sn11fe_valid]
        plt.plot(sn11fe_valid_phases,
                 sn11fe_valid_colors,
                 'ro', mfc='r', mec='r',
                 ms=MARKER_SIZE, alpha=SN11FE_PLOT_ALPHA)

        # plot original data points for 11fe
        temp, sn2011fe_orig_vmags = calc_v_band_mags(sn2011fe_red, zp)
        sn2011fe_orig = filter_lc( calc_colors(FILTER, sn2011fe_red, sn2011fe_orig_vmags, zp) )
        sn2011fe_orig_phases = [t[0] for t in sn2011fe_orig]
        sn2011fe_orig_colors = [t[1] for t in sn2011fe_orig]
        p1, = plt.plot(sn2011fe_orig_phases, sn2011fe_orig_colors, 'g+', mew=CROSS_SIZE)

        # format subplot
        if index%2 == 1:
            plt.ylabel("$V-X$")
        if index>2:
            plt.xlabel("Phase (relative B-max)")
        plt.ylim(row_ylims[FILTER])
        plt.xlim(LO,HI)
        ax.set_title(FILTER)
        index += 1

    # format figure
    fig2.suptitle("2012CU: Broadband Colors vs. Phase", fontsize=18)
    fig2.legend([sn11fe_plot_proxy, p1, p2],
                ["FTZ reddened 11fe: $E(B-V) = "+str(round(EBV_12CU,2))+"$, $R_V = "+str(round(BEST_RV,2))+"$",
                 "Original 11fe data",
                 "SNFactory 2012cu"],
                loc=3,
                ncol=3,
                mode="expand"
                )

    ############################################################################
    plt.show()

    

if __name__ == "__main__":
    main()
