import numpy as np
import matplotlib.pyplot as plt

rrs = np.loadtxt('paper/rrs.txt')
print(rrs)
ibis = 60/rrs

ts = np.cumsum(ibis)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts, rrs)
plt.show()
