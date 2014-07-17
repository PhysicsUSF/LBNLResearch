'''
::Author::
Andrew Stocker

::Description::
Test file for loader

::Last Modified::
07/16/2014

'''
import argparse
import loader as l
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from pprint import pprint


def main():
    # config argument parser
##    parser = argparse.ArgumentParser('Reddening law fitter for SN2011FE with respect to SN2012CU.')
##    parser.add_argument('lowcut', metavar='LO', default=None,
##                        help='the lower bound for the phase range to be graphed')
##    parser.add_argument('hicut', metavar='HI', default=None,
##                        help='the upper bound for the phase range to be graphed')
##    args   = parser.parse_args()
##    
##    lowcut = float(args.lowcut)
##    hicut  = float(args.hicut)

    # Function below is to filter out None values in the lightcurve and also filter by phase such that it is
    # in between the lowcut and hicut (inclusively).
    def lc_filter(lc, lowcut=None, hicut=None):
        if lowcut==None:
            lowcut = lc[0][0]
        if hicut==None:
            hicut = lc[-1][0]
        return filter(lambda t: t[1]!=None and (lowcut<=t[0]<=hicut), lc)

    zp     = l.load_filters()  # load filters in sncosmo registry and get zero-point fluxes


    #### Below I get both spectra from the loader and do the neccessary interpolation such that
    #### each list of spectra has matching phases.
    
    sn12cu = l.get_12cu()
    sn11fe = l.get_11fe()

    # filter out 12cu spectra with a phase outside of the phase range of 11fe spectra
    sn12cu = filter(lambda p: sn11fe[0][0]<=p[0]<=sn11fe[-1][0], sn12cu)
    # interpolate 11fe to 12cu phases
    sn11fe = l.interpolate_spectra( [t[0] for t in sn12cu], sn11fe )

    # these are the best guess values and other variables
    x0 = {'ebv':-1.37, 'rv':1.5, 'av':1.85, 'p':-2.1}
    redlaw = 'fm'
    filters = 'UBVRI'
    
    # least square best fit
    lsq_out, bmax_shifts = l.calc_lsq_fit(sn11fe, sn12cu, filters, zp, redlaw, x0)
    
    print "BMAXOUT:", bmax_shifts
    print "LSQOUT:", lsq_out
    
    RESULT = lsq_out[0]

    # get reddened 11fe based on lsq result and interpolate to 12cu phases
    sn11fe = l.get_11fe(redlaw, ebv=RESULT[0], rv=RESULT[1], av=RESULT[0], p=RESULT[1])
    sn11fe = l.interpolate_spectra([t[0] for t in sn12cu], sn11fe)

    ############################################################################

    
    plt.figure()
    for i, f in enumerate('UBVRI'):
        print f,'Plotting...'
        
        filter_name = 'tophat_' + f
        fcolor = plt.cm.gist_rainbow(i*25)  # color filter for plotting


        # compute 12cu magnitudes
        bandfluxes = zip([t[0] for t in sn12cu], [t[1].bandflux(filter_name) for t in sn12cu])
        bandfluxes = lc_filter(bandfluxes)

        phases = [t[0] for t in bandfluxes]
        bandmags = -2.5*np.log10( np.array([t[1] for t in bandfluxes])/zp[f] )

        p1, = plt.plot(phases, bandmags, 'o-', color=fcolor)


        # compute interpolated 11fe magnitudes
        bandfluxes = zip([t[0] for t in sn12cu], [t[1].bandflux(filter_name) for t in sn11fe])
        bandfluxes = lc_filter(bandfluxes)
        
        phases = [t[0] for t in bandfluxes]
        bandmags = -2.5*np.log10( np.array([t[1] for t in bandfluxes])/zp[f] )

        p2, = plt.plot(phases, bandmags+bmax_shifts[i], '^:', color=fcolor)

    plt.gca().invert_yaxis()
    plt.title('12CU, and 11FE reddened with E(B-V)='+str(RESULT[0])+', Rv='+str(RESULT[1]))
    plt.xlabel("Days after Bmax (MJD 56104.8)")
    plt.ylabel("Magnitude (Vega)")
    plt.legend([p1, p2], ['SN12CU','SN11FE'])
    plt.show()


if __name__ == "__main__":
    main()
