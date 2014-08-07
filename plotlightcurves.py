'''
::Author::
Andrew Stocker

::Description::
Program to plot the lightcurves of a reddened 11fe and 12cu for UBVRI
tophat filters.

::Last Modified::
07/23/2014

'''
import argparse
import loader as l
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from pprint import pprint

LOWCUT = -15.0
HICUT = 25.0
N_DAYS = 1.0
FILTERS = 'UBVRI'

################################################################################
##### FUNCTIONS ################################################################





# Function below is to filter out None values in the lightcurve and also filter by phase such that it is
# in between the lowcut and hicut (inclusively).
def lc_filter(lc, lowcut=None, hicut=None):
    if lowcut==None:
        lowcut = lc[0][0]
    if hicut==None:
        hicut = lc[-1][0]
    return filter(lambda t: t[1]!=None and (lowcut<=t[0]<=hicut), lc)


def get_ebv_bmax(sn11fe, sn12cu, zp):
    dist_mod_true = 5*np.log10(61./21.)
    s1, s2 = l.interpolate_spectra(0.0, sn11fe)[1], l.interpolate_spectra(0.0, sn12cu)[1]
    
    s1b, s1v = s1.bandflux('tophat_B'), s1.bandflux('tophat_V')
    s2b, s2v = s2.bandflux('tophat_B'), s2.bandflux('tophat_V')
    
    # add dist_mod_true to s1
    s1bmag = -2.5*np.log10( s1b/zp['B'] ) + dist_mod_true
    s1vmag = -2.5*np.log10( s1v/zp['V'] ) + dist_mod_true
    s2bmag = -2.5*np.log10( s2b/zp['B'] )
    s2vmag = -2.5*np.log10( s2v/zp['V'] )

    return (s2vmag-s1vmag) - (s2bmag-s1bmag) 


################################################################################
##### MAIN #####################################################################

def main():
    zp     = l.load_filters()  # load filters in sncosmo registry and get zero-point fluxes

    ##### INTERPOLATION AND LEAST SQUARE FITTING ###############################

    ## Below I get both spectra from the loader and do the neccessary interpolation such that
    ## each list of spectra has matching phases.
    
    sn12cu = l.get_12cu()
    sn11fe = l.get_11fe(loadptf=False, loadmast=False)

    # filter out 12cu spectra with a phase outside of the phase range of 11fe spectra
    sn12cu = filter(lambda p: LOWCUT<=p[0]<=HICUT, sn12cu)

    # find 12cu phases which are within 2 days of 11fe phases
    sn12cu_phases = np.array([t[0] for t in sn12cu])

    valid = find_valid([t[0] for t in sn11fe], sn12cu_phases, N_DAYS)
    nvalid = np.invert(valid)

    
    sn12cu_valid = [t for i, t in enumerate(sn12cu) if valid[i]]
    sn12cu_valid_phases = sn12cu_phases[valid]
    
    # interpolate 11fe to 12cu phases which are valid
    sn11fe_valid = l.interpolate_spectra(sn12cu_valid_phases, sn11fe)
    
    # these are the best guess values and other variables neccessary for scipy's leastsq
    ebv = get_ebv_bmax(sn11fe, sn12cu, zp)
    
    
    x0 = {'ebv':ebv, 'rv':1.5, 'av':1.85, 'p':-2.1}
    redlaw = 'fm'
    rv_guess = 1.5
    dist_mod_true = 5*np.log10(61./21.)
        
    dist_mod_weight = 1
    
    lsq_out = l.calc_ftz_lsq_fit(sn11fe_valid,
                                 sn12cu_valid,
                                 FILTERS,
                                 zp,
                                 ebv,
                                 rv_guess,
                                 dist_mod_true,
                                 dist_mod_weight,
                                 True,  # constrain distance
                                 False)  # weight distance modulus difference from true value

    # results
    RESULT = [ ebv, lsq_out[0][0] ]
    #dist_mod_shift = lsq_out[0][1]
    dist_mod_shift = dist_mod_true
    
    print
    print "### LEAST SQUARE FIT RESULTS ###"
    print "E(B-V)    :", ebv
    print "RV        :", RESULT[1]
    print "DMOD SHIFT:", dist_mod_shift
    print "*DMOD TRUE:", dist_mod_true
    print "################################"
    print 
    
    ############################################################################

    # get reddened 11fe based on lsq result and interpolate to 12cu phases
    sn11fe_orig = l.get_11fe(redlaw, ebv=RESULT[0], rv=RESULT[1])
    sn11fe = l.interpolate_spectra([t[0] for t in sn12cu], sn11fe_orig)

    # 12cu milky way extinction adjustment
    SN12CU_MW = dict( zip( 'UBVRI', [0.117, 0.098, 0.074, 0.058, 0.041] ))
    
    plt.figure()
    for i, f in enumerate(FILTERS):
        print f,'Plotting...'
           
        filter_name = 'tophat_' + f
        fcolor = plt.cm.gist_rainbow((3.0-'UBVRI'.index(f))/3.0)  # color filter for plotting

        # compute 12cu magnitudes
        bandfluxes = zip([t[0] for t in sn12cu], [t[1].bandflux(filter_name) for t in sn12cu])
        bandfluxes = lc_filter(bandfluxes) 

        phases = [t[0] for t in bandfluxes]
        bandmags = -2.5*np.log10( np.array([t[1] for t in bandfluxes])/zp[f] )

        p1, = plt.plot(phases, bandmags-SN12CU_MW[f], 'o-', color=fcolor)

        plt.text(26,bandmags[-1],f)  # write filter name next to respective line on plot


        # compute interpolated 11fe magnitudes
        bandfluxes = zip([t[0] for t in sn12cu], [t[1].bandflux(filter_name) for t in sn11fe])
        bandfluxes = lc_filter(bandfluxes)
        
        phases = np.array([t[0] for t in bandfluxes])
        bandmags = -2.5*np.log10( np.array([t[1] for t in bandfluxes])/zp[f] )

        # plot invalid interpolated phases as white diamonds
        p2, = plt.plot(phases, bandmags+dist_mod_shift, ':', color=fcolor)
        plt.plot(phases[valid], bandmags[valid]+dist_mod_shift, 'D', color=fcolor)
        p3, = plt.plot(phases[nvalid], bandmags[nvalid]+dist_mod_shift, 'D', color='w')


    plt.gca().invert_yaxis()
    titlestr = '12CU, 11FE reddened with FM: E(B-V)='+ str(round(RESULT[0],2)) \
                                                     + ', Rv=' + str(round(RESULT[1],2)) \
                                                     + ", DMOD_SHIFT: " + str(round(dist_mod_shift,2))
    plt.title(titlestr)
    plt.xlabel("Days after Bmax (MJD 56104.8)")
    plt.ylabel("Magnitude (Vega)")
    
    plt.legend([p1, p2, p3],
               ['12CU','11FE','not interpolated within\n'+str(N_DAYS)+' days (not used in fit)'],
               loc='lower left')
    plt.show()





if __name__ == "__main__":
    main()
