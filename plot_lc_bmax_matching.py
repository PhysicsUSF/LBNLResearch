'''
::Author::
Andrew Stocker

::Description::
Program to plot the lightcurves of 11fe and 12cu for UBVRI
tophat filters.

::Last Modified::
08/10/14

'''
import loader as l
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc


def main():
    filters = 'UBVRI'
    zp      = l.load_filters()
    prefix  = zp['prefix']
    
    # correct 12cu for Milky Way extinction (11fe already corrected)
    sn12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)
    sn12cu = filter(lambda t: -7<t[0]<28, sn12cu)
    sn12cu_phases = [t[0] for t in sn12cu]

    sn11fe = l.get_11fe()
    sn11fe = filter(lambda t: -7<t[0]<28, sn11fe)
    sn11fe_phases = [t[0] for t in sn11fe]

    
    fig, ax = plt.subplots(1)
    for i, f in enumerate(filters):
        print f,'Plotting...'
           
        filter_name = 'tophat_' + f
        fcolor = plt.cm.gist_rainbow((3.0-'UBVRI'.index(f))/3.0)  # color filter for plotting

        # compute and plot 12cu lightcurves
        sn12cu_bandmags = -2.5*np.log10( np.array([t[1].bandflux(prefix+f) for t in sn12cu])/zp[f] )
        p1, = plt.plot(sn12cu_phases, sn12cu_bandmags, 'o--', color=fcolor, mew=2, ms=11)

        # compute and plot 11fe lightcurves with bmax shift
        sn11fe_bandmags = -2.5*np.log10( np.array([t[1].bandflux(prefix+f) for t in sn11fe])/zp[f] )
        
        bmax_diff = np.min(sn12cu_bandmags)-np.min(sn11fe_bandmags)
        annotation = "{} (+{:.1f})".format(f, bmax_diff)
        plt.text(25, sn12cu_bandmags[-1]+0.4, annotation, family='monospace')
        
        p2, = plt.plot(sn11fe_phases,sn11fe_bandmags+bmax_diff, 'D:', color=fcolor, mew=1, ms=7)

        

    plt.gca().invert_yaxis()
    plt.xlabel("Days Relative to B-Maximum")
    plt.ylabel("Magnitude (Vega) + Offset")
    plt.title("SN2012CU and SN2011FE Lightcurves (Matched at B-Maximum)")
    plt.legend([p1, p2], ['sn2012cu','sn2011fe'], loc='lower left')
    plt.show()





if __name__ == "__main__":
    main()
