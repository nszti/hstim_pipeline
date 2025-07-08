import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt

imp_nanoz = ([9, 10, 15, 13, 13, 13, 13, 27, 12, 20, 13, 13, 14, 13, 19, 15, 17, 16, 15, 16, 15, 16, 19, 18, 17, 15, 15, 20, 13, 14, 14, 15, 13, 14, 22, 17, 17, 16, 19, 20, 19, 18, 16, 15, 18, 18, 17, 18, 12, 14, 14, 15, 14, 14, 18,20, 22, 17, 20, 18, 27, 36, 19, 16, 19, 40, 26, 24, 19, 21, 15, 16, 23, 18, 18 ])

imp_intan = ([8.78, 8.94, 13.1, 12.1, 12.4, 11.5, 11.9, 12.1, 12.3, 12.5, 13.2, 12.1, 23.4, 15.1, 16.1, 16, 15.7, 15.1, 15.6, 15.8, 18.6, 14.7, 14.1, 15.1, 15.2, 14.5, 14.6, 20.7, 16, 15.7, 16.9, 18.1, 20,16.6, 17.3, 21.1, 22.9, 21, 18.4, 15.5, 15.9, 18.9, 14.5, 14.2, 15.5, 14.6, 13.6,
              13.8, 18.6, 17.6, 18.2, 18.9, 25.7, 17.6, 18.1, 25.7, 20.3, 34, 44.2, 21.1, 19.7, 20.8, 20, 18.1, 26.9, 18.2, 16.3, 25.6,
              23.1, 28.7, 32.7, 53.1, 25.5, 29.8 ])



phase_nanoz = ([-15.4, -23.9, -20.5, -19.3, -18.6, -17.1, -29.1, -10, -16.1, -8.4, -19.9, -20.4, -21.1, -24.4, -16.8, -23.9, -15.1, -15.2, -13.9, -14.4, -14.5, -14.5, -26.3, -18.4, -17.5, -14.3, -15.9,
                -32.7, -16.8, -17, -15, -15.4, -18.2, -15.9, -23.3, -20.5, -27.1, -21.6, -21.9, -19.5, -18.9, -21.3, -21, -21.9, -19.9, -21.6, -24.2,
                -25.4, -28.9, -20.5, -21.6, -21.2, -23.4, -21.6, -27.7, -15.9, -23.4, -23.8, -18.6, -17.3, -13.9, -13.4, -18.3, -19.2, -26.4, -16.1, -13.9,
                -13.4, -18.5, -26.3, -14.8, -14.2, -15.1, -17.8, -14.3])

phase_intan = ([-25, -26, -23, -23, -22, -23, -34, -25, -24, -22, -25, -15, -6, -29, -19, -18, -17, -17, -17, -17, -30, -17, -18, -16, -15, -16, -16, -32, -14, -11, -13, -14, -4,-32, -27, -28, -23, -24, -25, -25, -25, -32, -24, -25, -23, -23, -21, -29, -26, -24, -20, -18, -12, -10, -28, -20, -22, -16, -15, -20, -21, -30, -18, -19, -14, -13, -13, -25, -14, -9, -11, -13, -1, -7 ])


t_stat, p_value = ttest_ind(phase_nanoz, phase_intan)
t_stat_2, p_value_2 = mannwhitneyu(phase_nanoz, phase_intan)

mean_nano = np.mean(phase_nanoz)
std_nano = np.std(phase_nanoz)

mean_intan = np.mean(phase_intan)
std_intan = np.std(phase_intan)

print(mean_nano, std_nano)
print(f"intant: {mean_intan}, {std_intan}")

print(f"t stat w t-test: {t_stat:.3f}")
print(f"p value {p_value:.3f}")
print(f"t stat w mann-whitney: {t_stat_2:.3f}")
print(f"p value {p_value_2:.3f}")

if p_value <0.05:
    print("szignifikans kulonbseg p< 0.05nel")

data = [imp_nanoz, imp_intan]
plt.boxplot(data, tick_labels = ['nanoz', 'intan'])
plt.ylabel('Imp')
x1 = np.random.normal(1,0.04, size=len(imp_nanoz))
x2 = np.random.normal(2,0.04, size=len(imp_intan))
plt.scatter(x1, imp_nanoz, marker = 'x', color = 'red', alpha = 0.7)
plt.scatter(x2, imp_intan, marker ='x', color = 'red', alpha = 0.7)

plt.show()