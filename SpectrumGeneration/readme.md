These four files have everything needed to generate neutron energy spectra. They all need to be in the same directory.

To obtain any spectra, do this:
1. Run "spectra_generation.py" with a python editor
2. In the python command line, execute the command "FRUIT" that was defined in "spectra_generation.py" by typing "spec = FRUIT(X)" where X is the number of spectra you want to generate.
3. The variable "spec" is now a numpy.ndarray that can be dumped to a csv or whatever you'd like.
4. To view any particular spectrum:

'''
import matplotlib.pyplot as plt
plt.semilogx(constants.Ebins, spec[X])
'''
