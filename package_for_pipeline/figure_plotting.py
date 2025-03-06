if stim_type == 'amp':
    legend = ['10', '20', '30', '40']
elif stim_type == 'freq':
    legend = ['50', '100', '200']
elif stim_type == 'pulse_dur':
    legend = ['50', '100', '200', '400']
else:
    legend = ['20', '50', '100', '200']
trialLabels = ['1', '2', '3', '4', '5']

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(legend, activeNeuronsPerBlock, marker="o")
axs[0, 0].set_xlabel('Stimulation current(uA)')
axs[0, 0].set_ylabel('Number of active neurons')
fig2, axs = plt.subplots(2, 2, figsize = (12,8))
axs[0, 0].plot(legend, avgCAperBlock, marker="o")
axs[0, 0].set_ylabel('Mean dF/F0')
axs[0, 0].set_xlabel('Stimulation amplitude (uA)')
axs[0, 1].plot(trialLabels, avgCAperTrial, marker="o")
axs[0, 1].set_ylabel('Mean dF/F0')
axs[0, 1].set_xlabel('Trial number')

plt.savefig(output_dir + '/fig.svg')