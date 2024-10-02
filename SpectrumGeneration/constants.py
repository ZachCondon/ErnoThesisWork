# McGreivy's original code had a lot more in this file, so it made more sense
#  for him to have a central file with constants he used in other scripts. It 
#  gives the option to be able to modify any constants once in this file rather
#  than in every file they're used in.

import numpy as np

# These energy bins are the energy structure I used in my thesis. The units are
#  eV (so right now it ranges from 1meV to 100MeV)
Ebins = np.array([1.00e-3, 1.58e-3, 2.51e-3, 3.98e-3, 6.31e-3, 1.00e-2, 1.58e-2, 2.51e-2, 3.98e-2, 6.31e-2, 1.00e-1, 1.58e-1, 2.51e-1, 3.98e-1, 6.31e-1, 1.00e-0, 1.58e-0, 2.51e-0, 3.98e-0, 6.31e-0, 1.00e+1, 1.58e+1, 2.51e+1, 3.98e+1, 6.31e+1, 1.00e+2, 1.58e+2, 2.51e+2, 3.98e+2, 6.31e+2, 1.00e+3, 1.58e+3, 2.51e+3, 3.98e+3, 6.31e+3, 1.00e+4, 1.58e+4, 2.51e+4, 3.98e+4, 6.31e+4, 1.00e+5, 1.26e+5, 1.58e+5, 2.00e+5, 2.51e+5, 3.16e+5, 3.98e+5, 5.01e+5, 6.31e+5, 7.94e+5, 1.00e+6, 1.12e+6, 1.26e+6, 1.41e+6, 1.58e+6, 1.78e+6, 2.00e+6, 2.24e+6, 2.51e+6, 2.82e+6, 3.16e+6, 3.55e+6, 3.98e+6, 4.47e+6, 5.01e+6, 5.62e+6, 6.31e+6, 7.08e+6, 7.94e+6, 8.91e+6, 1.00e+7, 1.12e+7, 1.26e+7, 1.41e+7, 1.58e+7, 1.78e+7, 2.00e+7, 2.51e+7, 3.16e+7, 3.98e+7, 5.01e+7, 6.31e+7, 7.94e7, 1.00e+8])

