import loader as l
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sncosmo as snc

from itertools import izip
from loader import redden_fm



def main():
        sn12cu = l.get_12cu('fm', ebv=1.0582, rv=2.5696)
        sn12cu = filter(lambda t: t[0]<28, sn12cu)
        phases = [t[0] for t in sn12cu]
        
        sn11fe = l.interpolate_spectra(phases, l.get_11fe())
        
        plt.figure()
        for i, t in enumerate(izip(sn12cu, sn11fe)):
                plt.plot(t[0][1].wave, t[0][1].flux-i*3e-15)
                plt.plot(t[1][1].wave, t[1][1].flux-i*3e-15)
        plt.show()
        
if __name__=="__main__":
        main()
