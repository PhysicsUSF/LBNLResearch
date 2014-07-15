import loader
import matplotlib.pyplot as plt

sndata = loader.SN2012CU

plt.figure()
for t in sndata:
    spectrum = t[1]
    plt.plot(spectrum.wave, spectrum.flux)
plt.show()
