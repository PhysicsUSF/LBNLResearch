'''
::Author::
Andrew Stocker

::Description::
This is a program for plotting a contour plot of the relative distance of sn12cu
and sn11fe.

::Last Modified::
07/23/2014

'''
import datetime
import loader as l
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
import sncosmo as snc

from loader import redden_fm
from pprint import pprint
from sys import argv


################################################################################
### FUNCTIONS ##################################################################
    
def find_valid(array,value_array,n):
    array = np.array(array)
    return_array = np.zeros(len(value_array), dtype=np.bool)
    for i, x in enumerate(value_array):
        idx = (np.abs(array-x)).argmin()
        return_array[i] = abs(array[idx]-x) <= n
    return return_array

def bandmags(f, spectra, zp):
    bandfluxes = [s.bandflux('tophat_'+f) for s in spectra]
    return -2.5*np.log10( bandfluxes/zp[f] )

def calc_chisq(rel_dmod, rv, ebv, sn11fe_spectra, ref, filters, zp):  
    reddened = [snc.Spectrum(spec.wave, redden_fm(spec.wave, spec.flux, ebv, rv)) for spec in sn11fe_spectra]
    reddened_mags = [bandmags(f, reddened, zp) for f in filters]

    new = np.concatenate( [mags + rel_dmod for mags in reddened_mags] )

    chisq = chisquare(new, ref)
    return chisq[0]

    #return np.sum((new-ref)**2)



################################################################################
################################################################################

def main():
    print
    ### args ###
    GEN_DATA = False if argv[1]=='False' else True
    
    ### vars ###
    EBV_12CU = -1.067
    EBV_14J  = -1.29
    
    LOWCUT = -15.0
    HICUT = 25.0
    N_DAYS = 2.0
    FILTERS = 'UBVRI'

    STEPS = 15

    ### generate meshs ###

    # 12cu
    x_12cu = np.linspace(2.0, 3.0, STEPS)  # relative distance modulus = 5.0*np.log10(ratio)
    y_12cu = np.linspace(1.0, 4.0, STEPS)  # R_V
    
    X_12CU, Y_12CU = np.meshgrid(x_12cu, y_12cu)
    Z_12CU = np.zeros( X_12CU.shape )

    # 14j
    x_14j = np.linspace(-2.0, -1.0, STEPS)  # relative distance modulus = 5.0*np.log10(ratio)
    y_14j = np.linspace(0.5, 3.5, STEPS)  # R_V
    
    X_14J, Y_14J = np.meshgrid(x_14j, y_14j)
    Z_14J = np.zeros( X_14J.shape )


    ############################################################################
    
    if GEN_DATA:
        
        ZP = l.load_filters()
        sn12cu_spectra = l.get_12cu()
        sn11fe_spectra = l.get_11fe(loadptf=False)
        sn14j          = l.get_14j()


# 12CU: ########################################################################


        ### Get 12cu magnitudes, get valid 12cu phases, get sn12cu reference mags ###
        print "GENERATING 12CU CHISQ MATRIX..."
        
        # milky way extinction for 12cu
        SN12CU_MW = dict( zip( FILTERS, [0.117, 0.098, 0.074, 0.058, 0.041] ))
        
        # filter out 12cu spectra with a phase outside of the phase range of 11fe spectra
        sn12cu_spectra = filter(lambda p: LOWCUT<=p[0]<=HICUT, sn12cu_spectra)

        # find 12cu phases which are within 2 days of 11fe phases
        sn12cu_phases = np.array([t[0] for t in sn12cu_spectra])
        sn12cu_spectra = [t[1] for t in sn12cu_spectra]
        valid = find_valid([t[0] for t in sn11fe_spectra], sn12cu_phases, N_DAYS)

        SN12CU_VALID_PHASES = sn12cu_phases[valid]
        SN12CU_REF_MAGS = np.concatenate( [bandmags(f, sn12cu_spectra, ZP) - SN12CU_MW[f] for f in FILTERS] )

        # get interpolated sn11fe reference spectra for reddening
        sn11fe_valid_spectra = l.interpolate_spectra(SN12CU_VALID_PHASES, sn11fe_spectra)
        SN11FE_REF_SPECTRA = [t[1] for t in sn11fe_valid_spectra]

        ########################################################################


        for i, rel_dmod in enumerate(x_12cu):
            for j, rv in enumerate(y_12cu):
                chisq = calc_chisq(rel_dmod, rv, EBV_12CU, SN11FE_REF_SPECTRA, SN12CU_REF_MAGS, FILTERS, ZP)

                print i, j, chisq
                Z_12CU[i,j] = chisq

        filename_12cu = "data/csv/" + datetime.datetime.now().strftime('12CU_lc_chisq_%Y-%m-%d_%H-%M-%S.csv')
        print
        print "SAVING TO:", filename_12cu
        print
        np.savetxt(filename_12cu, Z_12CU, delimiter=",")

# 14J:  ########################################################################


        ### Get 14j magnitudes, get valid 14j phases, get 14j reference mags ###
        print "GENERATING 14J CHISQ MATRIX..."

        sn14j_phases = np.array([-3.5, 5.2, 6.9, 7.9, 8.7, 16.7, 20.3, 28.1, 34.3])
        valid = find_valid([t[0] for t in sn11fe_spectra], sn14j_phases, N_DAYS)

        SN14J_VALID_PHASES = sn14j_phases[valid]

        sn11fe_valid_spectra = l.interpolate_spectra(SN14J_VALID_PHASES, sn11fe_spectra)
        SN11FE_REF_SPECTRA = [t[1] for t in sn11fe_valid_spectra]

        SN14J_REF_MAGS = []
        for f in FILTERS:
            FDATA = sn14j[f]

            mags = [(d['mag']-d['AX']) for d in FDATA if d['phase'] in SN14J_VALID_PHASES]

            SN14J_REF_MAGS.append( mags )

        SN14J_REF_MAGS = np.concatenate( SN14J_REF_MAGS )

        ########################################################################


        for i, rel_dmod in enumerate(x_14j):
            for j, rv in enumerate(y_14j):
                chisq = calc_chisq(rel_dmod, rv, EBV_14J, SN11FE_REF_SPECTRA, SN14J_REF_MAGS, FILTERS, ZP)

                print i, j, chisq
                Z_14J[i,j] = chisq

        filename_14j = "data/csv/" + datetime.datetime.now().strftime('14J_lc_chisq_%Y-%m-%d_%H-%M-%S.csv')
        print
        print "SAVING TO:", filename_14j
        print
        np.savetxt(filename_14j, Z_14J, delimiter=",")

    ############################################################################


    try:
        CHISQ_DATA_FILENAME_14J = argv[2]
    except:
        CHISQ_DATA_FILENAME_14J = filename_14j

    try:
        CHISQ_DATA_FILENAME_12CU = argv[3]
    except:
        CHISQ_DATA_FILENAME_12CU = filename_12cu


    print "LOADING 12CU FROM:", CHISQ_DATA_FILENAME_12CU
    loaded_12cu = np.loadtxt(open(CHISQ_DATA_FILENAME_12CU,"rb"), delimiter=",")
    Z_12CU = np.e**(-loaded_12cu/2.0)
    #Z_12CU = -np.log10( loaded_12cu )
    
    print "PLOTTING Z_12CU..."
    plt.figure()
    CS = plt.contour(X_12CU, Y_12CU, Z_12CU)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.ylabel('$R_V$')
    plt.xlabel('difference in host galaxy distance modulus (host_dist_mod - 11fe_dist_mod)')
    plt.title('SN2012CU')

    print
    print "LOADING 14J FROM:", CHISQ_DATA_FILENAME_14J
    loaded_14j = np.loadtxt(open(CHISQ_DATA_FILENAME_14J,"rb"), delimiter=",")
    Z_14J = np.e**(-loaded_14j/2.0)
    #Z_14J = -np.log10( loaded_14j )

    print "PLOTTING Z_14J..."
    plt.figure()
    CS = plt.contour(X_14J, Y_14J, Z_14J)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.ylabel('$R_V$')
    plt.xlabel('difference in host galaxy distance modulus (host_dist_mod - 11fe_dist_mod)')
    plt.title('SN2014J')

    
    plt.show()
    print



################################################################################
################################################################################
    
if __name__ == "__main__":
    main()
