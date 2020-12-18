import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from main import integrate, error
import scipy.stats as st

errors = {'max': [[], [], []], 'mean': [[], [], []], 'min': [[], [], []]}

keys = np.linspace(0, 100, )
full_data = pd.read_csv('predator-prey-data.csv')

for folder in os.listdir('reduced data/'):
    index = 0
    for file in os.listdir('reduced data/' + folder + '/'):
        data = pd.read_csv('reduced data/' + folder + '/' + file, header=None)
        for i in range(len(data)):
            x_val, y_val = integrate(np.array(data.iloc[i, 4:8]), full_data['t'], full_data['x'][0], full_data['y'][0], 
            'hoi', 'Joos')
            errors[folder][index] += [error(full_data['x'], full_data['y'], x_val, y_val, x_val, y_val)]

        index += 1



# fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

# ax1.boxplot(errors['max'], labels=['10%', '20%', '30%'])
# ax1.set_ylabel('MSE')
# ax1.set_ylim(0, 15)
# ax1.set_title('max')

# ax2.boxplot(errors['min'], labels=['10%', '20%', '30%'])
# ax2.set_ylabel('MSE')
# ax2.set_ylim(0, None)
# ax2.set_title('min')

# ax3.boxplot(errors['mean'], labels=['10%', '20%', '30%'])
# ax3.set_ylabel('MSE')
# ax3.set_title('mean')
# ax3.set_ylim(0, None)

# plt.subplots_adjust(hspace=0.3)
# plt.savefig('boxplot_removed data.pdf')
# plt.show()

for key, value in errors.items():
    for i in range(len(errors['max'])):
        st.ttest_ind(errors[key][i], errors[key][i-1])

all_errors = np.array([errors['max'], errors['mean'], errors['min']])
p_values = []
for l in range(3):
    for i in range(3):
        row = []
        for j in range(len(all_errors)):
            row += [round(st.ttest_ind(all_errors[l][i], all_errors[j][k])[1], 3) for k in range(len(all_errors))]
        p_values += [row]

p_values = np.round(np.array(p_values), 3)
print(p_values)

np.savetxt('p_values.csv', p_values, fmt='%.2e')