import loader as l
from pprint import pprint
import matplotlib.pyplot as plt

sn11fe = l.get_11fe(loadptf=True)

plt.figure()
for i in xrange(2, 5):
    plt.plot(sn11fe[i][1].wave, sn11fe[i][1].flux, label=str(sn11fe[i][0])+' '+str(sn11fe[i][2]))
plt.legend()
plt.show()







