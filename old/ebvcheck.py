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


def get_ebvs(sn11fe, sn12cu):
    if not np.array_equal([t[0] for t in sn11fe], [t[0] for t in sn12cu]):
        raise ValueError
    # requires not filters to be imported
    zp = l.load_filters('NOT_')
    prefix = zp['prefix']
    ebvs = []
    for i, p in enumerate([t[0] for t in sn11fe]):
        s1, s2 = sn11fe[i][1], sn12cu[i][1]  
        s1b, s1v = s1.bandflux(prefix+'B'), s1.bandflux(prefix+'V')
        s2b, s2v = s2.bandflux(prefix+'B'), s2.bandflux(prefix+'V')
        s1bmag = -2.5*np.log10( s1b/zp['B'] )
        s1vmag = -2.5*np.log10( s1v/zp['V'] )
        s2bmag = -2.5*np.log10( s2b/zp['B'] )
        s2vmag = -2.5*np.log10( s2v/zp['V'] )
        ebvs.append( (p, (s2bmag-s1bmag) - (s2vmag-s1vmag)) )
    return ebvs

        
def main():
    
    sn12cu  = l.get_12cu()
    sn11fe = l.get_11fe()
    
    phases = filter(lambda t: t<30, [t[0] for t in sn12cu])
    
    sn11fe = l.interpolate_spectra(phases, sn11fe)
    sn12cu = l.interpolate_spectra(phases, sn12cu)
    
    ebvs = get_ebvs(sn11fe, sn12cu)

    pprint( ebvs )
    print "avg:", np.average([t[1] for t in ebvs])

    plt.figure()
    plt.plot([t[0] for t in ebvs], [t[1] for t in ebvs])
    plt.show()

    
        
    
        
        

if __name__ == "__main__":
    main()
