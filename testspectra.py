import loader as l
from pprint import pprint
import matplotlib.pyplot as plt

sn11fe = l.get_11fe(loadptf=True)

pprint( [t[0] for t in sn11fe] )

plt.figure()
for i in xrange(10):
    plt.plot(sn11fe[i][1].wave, sn11fe[i][1].flux, label=str(sn11fe[i][0])+' '+str(sn11fe[i][2]))
plt.legend()
plt.show()







