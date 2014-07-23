'''
::Author::
Andrew Stocker

::Description::
Scripts to test our synthetic photometry calculations

::Last Modified::
07/21/2014

'''
import loader as l
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from pprint import pprint


def calc_ebvs(sn14j_colors, sn11fe_spectra, zp):
    # (14j vmag - 11fe vmag) - (14j bmag - 11fe bmag)

    ebvs = []
    for i, spectrum in enumerate(sn11fe_spectra):
        sn11fe_bmag = -2.5*np.log10( spectrum[1].bandflux('tophat_B')/zp['B'] )
        sn11fe_vmag = -2.5*np.log10( spectrum[1].bandflux('tophat_V')/zp['V'] )

        ebv = sn14j_colors[i] - (sn11fe_vmag - sn11fe_bmag)

        ebvs.append( (spectrum[0], ebv) )

    return ebvs

        
def main():
    zp     = l.load_filters()
    
    sn14j  = l.get_14j()
    sn11fe = l.get_11fe()

    dictlist_B = sn14j['B']
    
    phaselist_B = [d['phase'] for d in dictlist_B]

    colorlist_B = [(d['Vmag']-d['AV'])-(d['mag']-d['AX']) for d in dictlist_B]
    
    sn11fe = l.interpolate_spectra(phaselist_B, sn11fe)

    ebvs = calc_ebvs(colorlist_B, sn11fe, zp)

    print "AVERAGE E(B-V):", np.average( np.array([t[1] for t in ebvs]) )

    plt.figure()
    plt.plot([t[0] for t in ebvs], [t[1] for t in ebvs])
    plt.show()

    
        
    
        
        

if __name__ == "__main__":
    main()
