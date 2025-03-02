import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "./teacache_stats.npy"
data = np.load(file_path)

x = data[1]
y = data[3]
coefficients = np.polyfit(x, y, 4)
rescale_func = np.poly1d(coefficients)
ypred = rescale_func(x)
plt.clf()
plt.figure(figsize=(8,8))
plt.plot(np.log(x), np.log(y), '*',label='log residual output diff values',color='green')
plt.plot(np.log(x), np.log(ypred), '.',label='log polyfit values',color='blue')
plt.xlabel(f'log input_diff')
plt.ylabel(f'log residual_output_diff')
plt.ylim(-6,0)
plt.legend(loc=4) 
plt.title('4th order My Polynomial fitting ')
plt.tight_layout()
plt.savefig('residual_polynomial_fitting_log.png')
print(coefficients)