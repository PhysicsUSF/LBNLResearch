'''
::Author::
Andrew Stocker

::Description::
Test file for loader

::Last Modified::
07/16/2014
'''
import loader as l
import matplotlib.pyplot as plt
import numpy as np

def main():
    zp     = l.load_filters()
    sn12cu = l.get_12cu()

    interp_phase = 12.5
    
    plt.figure()
    plt.plot(sn12cu[5][1].wave, sn12cu[5][1].flux, label='phase: '+str(sn12cu[5][0]))
    plt.plot(sn12cu[6][1].wave, sn12cu[6][1].flux, label='phase: '+str(sn12cu[6][0]))
    interpd = l.interpolate_spectra( interp_phase, sn12cu )
    plt.plot(interpd[1].wave, interpd[1].flux, label='interp\'d at phase: '+str(interp_phase))
    plt.legend()
    plt.show()
    
    
##    plt.figure()
##    for sn in sndata:
##        plt.plot( sn[1].wave, sn[1].flux )
##    plt.show()

##    plt.figure()
##    for f in 'UBRVI':
##        filter_name = 'tophat_' + f
##        
##        bandfluxes = zip(PHASES, [t[1].bandflux(filter_name) for t in sndata])
##        bandfluxes = filter(lambda x: x[1]!=None, bandfluxes)
##
##        phases = [t[0] for t in bandfluxes]
##        bandmags = -2.5*np.log10( np.array([t[1] for t in bandfluxes])/zp[f] )
##        
##        plt.plot(phases, bandmags, 'o-', label=filter_name)
##
##    plt.gca().invert_yaxis()
##    plt.xlabel("Days after Bmax (MJD 56104.8)")
##    plt.ylabel("Magnitude (Vega)")
##    plt.legend()
##    plt.show()


if __name__ == "__main__":
    main()
