import numpy as np
import matplotlib.pyplot as plt

avgamp = [13.66666667, 63.66666667,	83.66666667,	99]
avgdur = [0.3333333333,	10.33333333,	66,	93.33333333]
num_block = 4
num_blockf = 3
trialNo = 5
trialLabels = [f"Trial {t + 1}" for t in range(trialNo)]
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# amp
axs[0,0].plot(range(num_block), avgamp, marker='o')
axs[0,0].set_xticks(range(num_block))
axs[0,0].set_title("Avg num of act neurons for amp")
axs[0,0].set_xlabel("Block")
axs[0,0].set_ylim(0,200)
axs[0,0].set_ylabel("Number active")

# dur
axs[0,1].plot(range(num_block), avgdur, marker='o')
axs[0,1].set_xticks(range(num_block))
axs[0,1].set_title("Avg num of act neurons for dur")
axs[0,1].set_xlabel("Block")
axs[0,1].set_ylim(0,160)
axs[0,1].set_ylabel("Number active")

avgfreq= [38.66666667,	55.33333333,	58.33333333]
num_blockf = 3
legend = [f"Block {b + 1}" for b in range(num_blockf)]  # stim param vals
# freq
axs[1,0].plot(range(num_blockf), avgfreq, marker='o')
axs[1,0].set_xticks(range(num_blockf))
axs[1,0].set_title("Avg num of act neurons for freq")
axs[1,0].set_xlabel("Bock")
axs[1,0].set_ylim(0,120)
axs[1,0].set_ylabel("Number active")

plt.tight_layout()
plt.show()
plt.close()

'''avgfreq= [38.66666667,	55.33333333,	58.33333333]
num_blockf = 3
trialNo = 5
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
legend = [f"Block {b + 1}" for b in range(num_blockf)]  # stim param vals
trialLabels = [f"Trial {t + 1}" for t in range(trialNo)]
# freq
axs.plot(range(num_blockf), avgfreq, marker='o')
axs.set_xticks(range(num_blockf))
axs.set_title("Avg num of act neurons for freq")
axs.set_xlabel("Bock")
axs.set_ylim(0,120)
axs.set_ylabel("Number active")
axs.legend()
plt.show()
plt.close()'''