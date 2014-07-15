import loader
import matplotlib.pyplot as plt
import numpy as np

def main():
    zp = loader.ZP_CACHE_VEGA
    sndata = loader.SN2012CU
    PHASES = np.array([t[0] for t in sndata])

    plt.figure()
    for f in 'UBRVI':
        filter_name = 'tophat_' + f
        
        bandfluxes = zip(PHASES, [t[1].bandflux(filter_name) for t in sndata])
        bandfluxes = filter(lambda x: x[1]!=None, bandfluxes)

        phases = [t[0] for t in bandfluxes]
        bandmags = -2.5*np.log10( np.array([t[1] for t in bandfluxes])/zp[f] )
        
        plt.plot(phases, bandmags, 'o-', label=filter_name)

    plt.gca().invert_yaxis()
    plt.xlabel("Days after Bmax (MJD 56104.8)")
    plt.ylabel("Magnitude (Vega)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
