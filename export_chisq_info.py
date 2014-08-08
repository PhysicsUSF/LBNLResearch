'''
Organization of Zach's Pickle:

[sn12cu_chisq_data] is a list of dictionaries for each phase.  Each dictionary has the form;

         {'BEST_AV' : 2.6129145970197083,
          'BEST_EBV': 0.92151898734177218,
          'BEST_RV' : 2.8354430379746836,
          'AV_1SIG' : (2.4570795246499628, 2.7612104769419865),
          'AV_2SIG' : (2.3874843882148093, 2.8683631548755382),
          'EBV_1SIG': (0.81518987341772153, 1.0354430379746837),
          'EBV_2SIG': (0.80000000000000004, 1.1113924050632913),
          'RV_1SIG' : (2.7215189873417724, 2.9303797468354431),
          'RV_2SIG' : (2.6455696202531644, 3.0063291139240507),
          'phase': 23.5,
          'x':  array([ 0.8       ,  0.80759494,  0.81518987,  0.82278481,  0.83037975,
                        0.83797468,  0.84556962,  0.85316456,  0.86075949,  0.86835443,
                        ...,
                        1.36962025,  1.37721519,  1.38481013,  1.39240506,  1.4       ]),
          'y':  array([ 2.        ,  2.01898734,  2.03797468,  2.05696203,  2.07594937,
                        2.09493671,  2.11392405,  2.13291139,  2.15189873,  2.17088608,
                        ...,
                        3.42405063,  3.44303797,  3.46202532,  3.48101266,  3.5       ])
          
          'CDF': array([[ 1.,  1.,  1., ...,  1.,  1.,  1.],
                        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
                        ..., 
                        [ 1.,  1.,  1., ...,  1.,  1.,  1.]]),
          }
          
where [x] and [y] are the values used for EBV and RV respectively, and CDF is the
cumulative distribution function calculated by (dof=18):

        CHISQ = (dof/Zmin)*Z
        CDF = 1-exp(-CHISQ/2)
        
'''
import pickle
from pprint import pprint

from plot_excess_contours import get_12cu_best_ebv_rv

SN12CU_CHISQ_DATA = get_12cu_best_ebv_rv()

pickle.dump(SN12CU_CHISQ_DATA, open('sn12cu_chisq_data.pkl', 'wb'))

f = pickle.load(open('sn12cu_chisq_data.pkl', 'rb'))
pprint( f )
