These four files have everything needed to generate neutron energy spectra. They all need to be in the same directory.

To obtain any spectra, do this:
1. Run "spectra_generation.py" with a python editor
2. To generate spectra, where X is the number of spectra you want to generate:
```
spec = FRUIT(X)
```
4. The variable "spec" is now a numpy.ndarray that can be dumped to a csv or whatever you'd like.
5. To quickly view any particular spectrum in a figure:
```
import matplotlib.pyplot as plt
plt.semilogx(constants.Ebins, spec[X])
```

<!-- COMMENT -->
When you want to modify this code to be just high energy, all you should need to do is open up the constants folder and redefine Ebins so that it covers the energy range you want.
