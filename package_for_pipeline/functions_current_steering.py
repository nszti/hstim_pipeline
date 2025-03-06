import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from pathlib import Path
import ast


#frameNos
def frameNos_val(frameNos, expDir):
    '''
    :param frameNos: frame numbers, np.array, example: np.array([1085, 1085, 1085, 1085])
    :return:
    '''
    '''
    for dir in os.listdir(expDir):
        #print(frameNos.shape)
        #print(frameNos)
        np.save(expDir + '/' + dir + '/frameNos.npy', frameNos)
    '''
#stim_dur
def stim_dur_val(stim_duration, expDir):
    '''
    :param stimDuration: stim duration time, np.array, example: np.array([1, 1, 1, 1])
    :return:
    '''

    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for dir in filenames:
        #print(stim_duration.shape)
        #np.save(expDir + '/' + dir + '/stimDurations.npy', stim_duration)
        dir_path = os.path.join(base_dir, dir)
        np.save(dir_path +'/stimDurations.npy', stim_duration)

#stim_time
def stim_time_val(start_time, expDir):
    #2d arrayt tudjon kapni, ,fuggvenyen belul loopolni?
    '''
    :param start_time: stim start timepoint, np.array, example: np.array([336, 323, 319, 329])
    :return:
    '''
    '''
    for dir in os.listdir(expDir):
        #print(start_time)
        np.save(expDir + '/' + dir + '/stim_start_times.npy', start_time)
    '''
#electrodeROI
def electROI_val(electrodeROI, csteering = False):
    '''
    :param electrodeROI: number, example: np.array(electrodeROIs[8])
    :return:
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for dir in filenames:
        if csteering:
            if len(electrodeROI) !=2:
                raise ValueError("Current steering is True, needs 2 electrode number")
            else:
                dir_path = os.path.join(base_dir, dir)
                np.save(dir_path + '/electrodeROI.npy', electrodeROI)
        else:
            #print(electrodeROI)
            dir_path = os.path.join(base_dir, dir)
            np.save(dir_path + '/electrodeROI.npy', electrodeROI)
            #np.save(expDir + '/' + dir + '/electrodeROI.npy', electrodeROI)
#baseline
def baseline_val(expDir, csteering = False ):
    '''
    :param expDir: example: 'c:/Hyperstim/Deliverable/GCaMP6f/'
    :return: saves all_norm_traces, prints shape of all_norm_traces, output: F0.npy
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]

    for dir in filenames:
        #print(dir)

        F_path = expDir + '/' + dir + '/suite2p/plane0/F.npy'
        iscell_path = expDir + '/' + dir + '/suite2p/plane0/iscell.npy'
        stim_start_times_path = expDir + '/' + dir + '/stimTimes.npy'
        F = np.load(F_path, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)
        stim_start_times = np.load(stim_start_times_path, allow_pickle=True)
        #print(f"stim type: {type(stim_start_times)}")
        #frames =

        if csteering:
        else:

            # Define baseline duration
            baseline_duration = int(stim_start_times[0][0])-1  # Duration in milliseconds   #je ne sais pas ce qui c'est passé ici
            #print(baseline_duration)

            # create empty list to store normalized baseline_diffs
            all_norm_traces = []
            cellcount = 0
            '''
            for batch in stim_start_times:
                first_value = batch[0]
                baseline_duration = int(first_value)-1 #!!int()
            '''
            # Iterate through all rois
            for cell_index, (fluorescence_trace, (iscell_value, _)) in enumerate(zip(F, iscell)):
                #print(fluorescence_trace)
                # Check iscell==1
                if iscell_value == 1:
                    cellcount += 1
                    baseline_value = np.mean(fluorescence_trace[:baseline_duration])
                    normalized_trace = (fluorescence_trace - baseline_value) / baseline_value
                    #plt.plot(normalized_differences)
                    #plt.show()
                    all_norm_traces.append(normalized_trace)

            # convert the list of baseline_diffs to a npy array
            all_norm_traces = np.array(all_norm_traces)

            #save output as .npy file
            dir_path = os.path.join(base_dir, dir)
            np.save(dir_path + '/suite2p/plane0/F0.npy', all_norm_traces)
            print(f"F0.npy saved to {dir_path + '/suite2p/plane0/F0.npy'}")
            #print(all_norm_traces.shape)
            #print(all_norm_traces)

#activated_neurons
def activated_neurons_val(expDir):
    '''
    :param input_file_path: 'D:/2P/E/test/merged_GCaMP6f_23_09_25_3-6_pos_amp/'
    :param time_block: type: number, time block duration in frames, example: 1085
    :return: saves distance results as 'result_df', can print sum of roi_num-med_val distances, output: activated_neurons.npy
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for i, dir in enumerate(filenames):
        # Load the fluorescence traces and iscell array
        F0_path = expDir + '/' + dir + '/suite2p/plane0/F0.npy'
        iscell_path = expDir + '/' + dir + '/suite2p/plane0/iscell.npy'
        ROI_numbers_path = expDir + '/' + dir + '/suite2p/plane0/ROI_numbers.npy'
        stim_start_times_path = expDir + '/' + dir + '/stimTimes.npy'
        frame_numbers_path = expDir + '/' + dir + '/frameNum.npy'

        F0 = np.load(F0_path, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)
        ROI_numbers = np.load(ROI_numbers_path, allow_pickle=True)
        stim_start_times = np.load(stim_start_times_path, allow_pickle=True)
        frame_numbers = np.load(frame_numbers_path, allow_pickle=True)
        #print(stim_start_times)
        if i < len(frame_numbers):
            time_block = int(frame_numbers[i]) #1085
        if i < len(stim_start_times):
            #print(stim_start_times)
            baseline_duration = int(stim_start_times[i][0])
            #print(baseline_duration)
        # Calculate TIFF trigger start and end tuples
        num_tif_triggers = int(len(F0[0]) / time_block)
        tif_triggers = []

        for i in range(num_tif_triggers):
            start_time = i * time_block
            end_time = start_time + time_block
            tif_triggers.append((start_time, end_time))

        # Define baseline duration
        baseline_duration = int(stim_start_times[



















                                ][0]) # Duration in milliseconds
        #print(baseline_duration)
        '''
        for batch in stim_start_times:
            first_value = stim_start_times[batch]
            baseline_duration = int(first_value)
            #print(baseline_duration)
        '''
        baseline_durs=[]
        # Create an empty list to store threshold values for each ROI and tuple
        threshold_list = []
        # Create an empty list to store results for each ROI and tuple
        results_list = []
        # Iterate through all ROIs
        for i in range(len(F0)):
            roi_thresholds = []
            roi_results = []
            # Iterate through all TIFF triggers
            for tif_trigger in tif_triggers:
                # Extract start and end time stamps for the current tuple
                start_time, end_time = tif_trigger
                #print(f"stimstarttime {i}type: {type(stim_start_times[i][0])} ")
                #print(f"start: {start_time}, end: {start_time + baseline_duration}, baseline: {baseline_duration}, f0: {F0.shape}")
                #start_time = start_time[0] if isinstance(start_time, list) else start_time
                #end_time = end_time[0] if isinstance(end_time, list) else end_time
                # Create lists to store threshold and results for the current ROI
                #print(f"start: {start_time}, end: {start_time + baseline_duration}, baseline: {baseline_duration}, f0: {F0.shape}")
                baseline_dur = F0[i,start_time:start_time + baseline_duration] #??  #je ne sais pas ce qui c'est passé ici

                # Calculate average for baseline
                baseline_avg = np.mean(baseline_dur)
                #print(baseline_avg)
                # Calculate standard deviation for the baseline trace
                baseline_std = np.std(baseline_dur)
                # Calculate threshold for the current tuple
                threshold = baseline_std * 1 + baseline_avg
                # Append threshold to the list for the current ROI
                roi_thresholds.append(threshold)
                # Check if fluorescence exceeds threshold for the current tuple
                stim_avg = np.mean(F0[i,(start_time + baseline_duration):(start_time + baseline_duration + 465)])
                if stim_avg > threshold:
                    exceed_threshold = 1
                else:
                    exceed_threshold = 0
                # Append result (1 or 0) to the list for the current ROI
                roi_results.append(int(exceed_threshold))
            # Append threshold values and results for the current ROI to the overall lists
            threshold_list.append(roi_thresholds)
            results_list.append(roi_results)
            #print(threshold_list)
            #print(roi_results)

        # Convert the lists of threshold values and results to NumPy arrays---???not used
        threshold_array = np.array(threshold_list)
        results_array = np.array(results_list)
        result_df = pd.DataFrame({
            'ROI_number': ROI_numbers,
            #'thresholds': threshold_list,
            'activated_neurons': results_list
        })
        pd.set_option('display.max_rows', None)
        '''
        catSum = []
        for j in range(4):
            cat = []
            for i in range(len(results_list)):
                cat.append(results_list[i][j])
            catSum.append(sum(cat))
        '''

        #print(catSum)
        dir_path = os.path.join(base_dir, dir)
        np.save(dir_path + '/suite2p/plane0/activated_neurons.npy', result_df)
        print(f"activated_neurons.npy saved to {dir_path + '/suite2p/plane0/activated_neurons.npy'}")
        #np.save(expDir + '/' + dir + '/suite2p/plane0/activated_neurons.npy', result_df)

#timecourse
def timecourse_vals(expDir, frame_rate, num_trials):
    '''
    :param expDir:
    :param frame_rate: 31Hz
    :param num_trials: 5
    :return: saves 'results.npz'
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for dir in filenames:
        #print(expDir + '/' + dir)
        F_path = expDir + '/' + dir + '/suite2p/plane0/F0.npy'
        stim_start_times_path = expDir + '/' + dir + '/stimTimes.npy'
        stim_duration_path = expDir + '/' + dir + '/stimDurations.npy'
        block_frames_path =expDir + '/' + dir + '/frameNum.npy'
        roi_number_path = expDir + '/' + dir + '/suite2p/plane0/ROI_numbers.npy'

        F = np.load(F_path, allow_pickle = True)
        stim_start = np.load(stim_start_times_path,allow_pickle=True)
        block_frames = np.load(block_frames_path,allow_pickle=True)
        stim_duration = np.load(stim_duration_path, allow_pickle = True)
        roi_num = np.load(roi_number_path, allow_pickle = True)
        start_timepoints = []
        for i in stim_start:
            start_timepoints.append(i)

        time_block = []
        for b in block_frames:
            time_block.append(b)
        stimulation_duration = []
        for s in stim_duration:
            stimulation_duration.append(s)

        num_blocks = len(time_block)

        resting_period = 2
        rest_dur_f = resting_period * frame_rate
        stim_dur_f = []
        end_f =[]

        for s in stimulation_duration:
            frameNo = math.floor(s*frame_rate)
            stim_dur_f.append(frameNo)
            end_f.append(frameNo + rest_dur_f)

        blocks_start = []
        for i in range(len(time_block)):
            prev_blocks_duration = sum(time_block[0:i])
            start_time = prev_blocks_duration
            end_time = start_time + time_block[i]-1
            blocks_start.append(start_time)

        start_stim_times = []
        trial_start_t = []
        trial_end_t = []
        trial_stim_end_t = []
        threshold_block = []
        rest_traces = []
        stim_traces = []
        avgs_stim = []
        avgs_rest = []
        exceed_thr = []
        stimResults = np.empty([len(F), num_blocks, num_trials],'int')
        restResults = np.empty([len(F), num_blocks, num_trials],'int')
        stimAvgs = np.empty([len(F), num_blocks, num_trials])
        restAvgs = np.empty([len(F), num_blocks, num_trials])
        baselineAvgs = np.empty([len(F), num_blocks])

        for iTrace in range(len(F)):
            stim_result_list =[]
            rest_result_list = []
            for iBlock  in range(num_blocks):
                baseline_dur = F[iTrace, int(blocks_start[iBlock]): int(blocks_start[iBlock]) + (int(start_timepoints[iBlock])-1)]
                baseline_avg = np.mean(baseline_dur)
                baselineAvgs[iTrace,iBlock] = baseline_avg
                baseline_std = np.std(baseline_dur)
                threshold = baseline_std * 3 + baseline_avg

                avgs_stim_trial = []
                avgs_rest_trial = []
                for iTrial in range(num_trials):
                    trial_start = blocks_start[iBlock] + (int(start_timepoints[iBlock]) + iTrial * int(end_f[iBlock]))
                    trial_end = int(trial_start) + stim_dur_f[iBlock]
                    #print(f"trial start: {trial_start}, stimdur: {stim_dur_f[iBlock]}")
                    #print(f"itrace: {iTrace}, trial start: {trial_start}, trial_end: {trial_end}")
                    stim_trace = F[iTrace, int(trial_start):int(trial_end)]
                    avg_stim = np.mean(stim_trace)
                    stimAvgs[iTrace][iBlock][iTrial] = avg_stim

                    if avg_stim > threshold:
                        stim_above_thr = True
                    else:
                        stim_above_thr = False

                    stimResults[iTrace][iBlock][iTrial] = stim_above_thr

                    rest_trace_start = blocks_start[iBlock] + (int(start_timepoints[iBlock]) + ((iTrial+1) * (int(stim_dur_f[iBlock])) + (int(iTrial*rest_dur_f))))
                    rest_trace_end = int(rest_trace_start) + int(rest_dur_f)
                    rest_trace = F[iTrace, int(rest_trace_start):int(rest_trace_end)]
                    avg_rest = np.mean(rest_trace)
                    restAvgs[iTrace][iBlock][iTrial] = avg_rest

                    if avg_rest > threshold:
                        rest_above_thr = True

                    else:
                        rest_above_thr = False
                    restResults[iTrace, iBlock, iTrial] = rest_above_thr

        numRows = math.ceil(math.sqrt(len(F)))
        fig, axs = plt.subplots(numRows, numRows, squeeze= False)

        full_trial_traces = np.zeros((len(F), num_blocks, num_trials,124), dtype=object)
        for iTrace in range(len(F)):
            for iBlock in range(num_blocks):
                for iTrial in range(num_trials):
                    full_trial = np.concatenate((stim_trace, rest_trace))
                    full_trial_traces[iTrace, iBlock, iTrial,0:len(full_trial)] = full_trial

        for i in range(numRows):
            for j in range(numRows):
                if i*numRows+j < len(F):
                    axs[i][j].imshow(stimResults[i*numRows+j,:,:])
                    axs[i][j].set_title('ROI' + str(roi_num[i*numRows+j]))
                else:
                    print()
        #plt.show()

        np.savez(expDir + '/' + dir + '/results.npz', stimResults=stimResults, restResults=restResults, stimAvgs=stimAvgs, restAvgs=restAvgs, baselineAvgs=baselineAvgs, full_trial_traces=full_trial_traces)
        print(f"results.npz saved to {expDir + '/' + dir + '/results.npz'}")
#distance
def dist_vals (expDir):
    '''
    :param:
    :return:
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for dir in filenames:

        iscell_path = expDir + '/' + dir + '/suite2p/plane0/iscell.npy'
        stat_path = expDir + '/' + dir + '/suite2p/plane0/stat.npy'
        electrodeROI_path = expDir + '/' + dir + '/electrodeROI.npy'

        stat = np.load(stat_path, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)
        electrodeROI = np.load(electrodeROI_path, allow_pickle=True)

        #extract cell roi info
        first_column = iscell[:, 0]
        tempforcells = []
        for index, value in enumerate(first_column):
            if value ==1:
                roi_info = f"{index}, Value: {value}"
                tempforcells.append([roi_info.split(',')[0]])


        #extract all roi med info
        med_values = [roi['med']for roi in stat]
        tempforallmed = []
        tempforallroi = []
        for roi_number, med_value in enumerate(med_values):
            tempforallroi.append(roi_number)
            tempforallmed.append(med_value)

        #dataframes for cells & all roi
        dfcell_roi = pd.DataFrame(tempforcells, columns = ['roi_num'])
        mergedallmedinfo = list(zip(tempforallroi, tempforallmed))
        dfallmedinfo = pd.DataFrame(mergedallmedinfo, columns=['roi_num', 'med'])

        #matching
        matched_roi_med = []
        for roi_num in tempforcells:
            roi_num = int(roi_num[0])  # extracting roi nums from tempforcells
            if roi_num in tempforallroi:
                med_value = dfallmedinfo.loc[dfallmedinfo['roi_num'] == roi_num, 'med'].values
                if len(med_value) > 0:
                    matched_roi_med.append((roi_num, med_value[0]))
        #df for matched info
        dfmatched = pd.DataFrame(matched_roi_med, columns = ['roi_num', 'med_value'])

        # Distance calc w dfmatched-------------------------------------------
        def euclidean_distance(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))
        # fv minimum distance search
        def minimum_distance_search(med_values, start_roi):
            start_point = None  #spec starting point(ha kell)
            for roi, coords in zip(roi_numbers, med_values):
                if roi == start_roi:
                    start_point = coords
                    break

            if start_point is None:
                raise ValueError(f"ROI {start_roi} not found in the dataset.")
            distances = [euclidean_distance(start_point, coords) for coords in med_values]
            return distances

        roi_numbers = dfmatched['roi_num']
        med_values =dfmatched['med_value']
        distances = minimum_distance_search(med_values, electrodeROI)
        #print(distances)
        # extracting electrode info
        electrode_i = dfmatched[dfmatched['roi_num'] == electrodeROI].index
        electrode_med= dfmatched.loc[electrode_i, 'med_value'].iloc[0]
        x_value,y_value = electrode_med
        #print(distances)

        # Distance calc w dfmatched-------------------------------------------

        #df for electrode med info
        electrode_df = pd.DataFrame({'electrode med x': [x_value], 'electrode med y': [y_value]})

        #Results
        result_df = pd.DataFrame({
            'ROI_Number': roi_numbers,
            'Med_Values': med_values,
            'distance': distances
        })

        #print(result_df)
        #print(result_df.shape)
        dir_path = os.path.join(base_dir, dir)
        np.save(dir_path + '/suite2p/plane0/distances.npy', result_df)
        np.save(dir_path + '/suite2p/plane0/ROI_numbers.npy', roi_numbers)
        print(f"distances.npy saved to {dir_path + '/suite2p/plane0/distances.npy'}, ROI_numbers.npy saved to {dir_path + '/suite2p/plane0/ROI_numbers.npy'}")
        # save output as .npy file
        #np.save(expDir + '/' + dir + '/suite2p/plane0/distances.npy', result_df)
        #np.save(expDir + '/' + dir + '/suite2p/plane0/distances.npy', distances)
        #np.save(expDir + '/' + dir + '/suite2p/plane0/ROI_numbers.npy', roi_numbers)

#data_analysis
def data_analysis_values (stim_type, expDir):
    '''
    :param stim_type: 4 type: 'amp','freq','pulse_no','dur'
    :return:
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for dir in filenames:
        #load files and data
        container = np.load(expDir + '/' + dir + '/results.npz', allow_pickle=True)
        #print(container)
        distances = np.load(expDir + '/' + dir + '/suite2p/plane0/distances.npy', allow_pickle=True)
        #print(distances)
        ROI_IDs = np.load(expDir + '/' + dir + '/suite2p/plane0/ROI_numbers.npy', allow_pickle=True)
        #print(ROI_IDs)
        electrode_ROI = np.load(expDir + '/' + dir + '/electrodeROI.npy', allow_pickle=True)
        distanceFromElectrode = distances[:, 2]
        #print(distanceFromElectrode)
        stimResults = container["stimResults"]
        restResults = container["restResults"]
        stimAvgs = container["stimAvgs"]
        restAvgs = container["restAvgs"]
        baselineAvgs = container["baselineAvgs"]
        baselineAvgs = container["baselineAvgs"]
        full_trial_traces = container["full_trial_traces"]
        #print(full_trial_traces.shape)

        #remove electrode ROI from data
        electrode_ROI_index = np.where(ROI_IDs == electrode_ROI)[0]
        distanceFromElectrode=np.delete(distanceFromElectrode, electrode_ROI_index, axis=0)
        stimResults= np.delete(stimResults, electrode_ROI_index, axis=0)
        restResults = np.delete(restResults, electrode_ROI_index, axis=0)
        stimAvgs = np.delete(stimAvgs, electrode_ROI_index, axis=0)
        restAvgs = np.delete(restAvgs, electrode_ROI_index, axis=0)
        baselineAvgs = np.delete(baselineAvgs, electrode_ROI_index, axis=0)
        full_trial_traces = np.delete(full_trial_traces, electrode_ROI_index, axis=0)


        #collect ROI, block and trial numbers
        ROI_No = stimResults.shape[0]
        block_No = stimResults.shape[1]
        trial_No = stimResults.shape[2]

        if stim_type == 'amp':
            legend = ['10','20']
        elif stim_type == 'freq':
            legend = ['50', '100', '200']
        elif stim_type == 'pulse_no':
            legend = ['20','50', '100', '200']
        else:
            legend = ['50', '100', '200', '400']
        trialLabels = ['1', '2', '3', '4', '5']

        #collect neurons activated during a block
        activatedNeurons = np.empty([ROI_No, block_No],'int')
        for iROI in range(ROI_No):
            for iBlock in range(block_No):
                sumTrials = sum(stimResults[iROI, iBlock, :])
                if sumTrials > 0:
                    activatedNeurons[iROI][iBlock] = 1
                else:
                    activatedNeurons[iROI][iBlock] = 0

        #compute the number and fraction of neurons activated (or silent) during a block
        activeNeuronsPerBlock = np.empty(block_No,'int')
        silentNeuronsPerBlock = np.empty(block_No,'int')
        activeNeuronsPerBlockFraction = np.empty(block_No)
        silentNeuronsPerBlockFraction = np.empty(block_No)

        for iBlock in range(block_No):
            activeNeuronsPerBlock[iBlock] = sum(activatedNeurons[:,iBlock])
            activeNeuronsPerBlockFraction[iBlock] = activeNeuronsPerBlock[iBlock]/ ROI_No
            silentNeuronsPerBlock[iBlock] = stimResults.shape[0] - activeNeuronsPerBlock[iBlock]
            silentNeuronsPerBlockFraction[iBlock] = silentNeuronsPerBlock[iBlock] / ROI_No

        #plot the number and fraction of neurons activated (or silent) during a block
        fig, axs = plt.subplots(2, 2)
        axs[0,0].plot(legend, activeNeuronsPerBlock, marker="o")
        axs[0,0].set_xlabel('Stimulation current(uA)')
        axs[0,0].set_ylabel('Number of active neurons')


        axs[0,1].plot(legend, activeNeuronsPerBlockFraction, marker="o")
        axs[0,1].set_xlabel('Stimulation current(uA)')
        axs[0,1].set_ylabel('Fraction of active neurons')
        #plt.savefig(output_path + '/test.svg')


        #compute the number and fraction of neurons activated during trials of a block
        activeNeuronsPerBlockPerTrial = np.empty([trial_No, block_No],'int')
        activeNeuronsPerBlockPerTrialFraction = np.empty([trial_No, block_No])

        for iBlock in range(block_No):
            for iTrial in range(trial_No):
                activeNeuronsPerBlockPerTrial[iTrial][iBlock] = sum(stimResults[:, iBlock, iTrial])
                activeNeuronsPerBlockPerTrialFraction[iTrial][iBlock] = sum(stimResults[:, iBlock, iTrial]) / ROI_No


        #plot the number and fraction of neurons activated during trials of a block
        axs[1,0].plot(trialLabels, activeNeuronsPerBlockPerTrial, marker="o")
        axs[1,0].legend(legend)

        axs[1,1].plot(trialLabels, activeNeuronsPerBlockPerTrialFraction, marker="o")
        axs[1,1].legend(legend)


        #calculate and plot the mean amplitudes during stimulation trials and blocks
        avgCA = np.empty([block_No, trial_No])
        avgCAperTrial = np.empty([trial_No])
        avgCAperBlock = np.empty([block_No])
        for iBlock in range(block_No):
            for iTrial in range(trial_No):
                avgCA[iBlock][iTrial] = np.mean(stimAvgs[:, iBlock, iTrial])
        avgCAperTrial = np.mean(avgCA, axis = 0)
        avgCAperBlock = np.mean(avgCA, axis = 1)

        fig2, axs = plt.subplots(2, 2)
        axs[0,0].plot(legend, avgCAperBlock, marker="o")

        axs[0,1].plot(trialLabels, avgCAperTrial, marker="o")

        #calculate and plot the mean amplitudes during stimulation trials of a block
        avgCAduringTrials = np.empty([block_No, trial_No])
        for iBlock in range(block_No):
            for iTrial in range(trial_No):
                avgCAduringTrials[iBlock][iTrial] = np.mean(stimAvgs[:, iBlock, iTrial])

            axs[1,0].plot(trialLabels, avgCAduringTrials[iBlock,:])
        axs[1,0].legend(legend)

        #calculate and plot the mean amplitudes during rest periods of a block
        avgCAduringRest = np.empty([block_No, trial_No])
        for iBlock in range(block_No):
            for iTrial in range(trial_No):
                avgCAduringRest[iBlock][iTrial] = np.mean(restAvgs[:, iBlock, iTrial])

            axs[1,1].plot(trialLabels, avgCAduringRest[iBlock,:])
        axs[1,1].legend(legend)

        #plot calcium traces during stimulation
        tracesPerBlock = np.empty([ROI_No, 124])
        avgTracePerBlock = np.empty([block_No, trial_No, 124])
        for iBlock in range(block_No):
            for iTrial in range(trial_No):
                tracesPerBlock = full_trial_traces[:,iBlock,iTrial,:]
                avgTracePerBlock[iBlock, iTrial,:] = np.mean(tracesPerBlock, axis = 0) #

        plot_dur = 3*31
        ymin = -0.01
        ymax = 0.2
        fig3, axs = plt.subplots(2,5)
        for iBlock in range(block_No):
            for iTrial in range(trial_No):
                axs[0, iBlock].plot(avgTracePerBlock[iBlock, iTrial, 0:plot_dur])
                axs[0, iBlock].set_title(legend[iBlock])
                axs[0, iBlock].set_ylim([ymin, ymax])
                axs[0, iBlock].legend(trialLabels)
                axs[1, iTrial].plot(avgTracePerBlock[iBlock, iTrial, 0:plot_dur])
                axs[1, iTrial].set_title(trialLabels[iTrial])
                axs[1, iTrial].set_ylim([ymin, ymax])
                axs[1, iTrial].legend(legend)

        #distance calculation and plot
        binSize = 50
        maxDistance = 600
        bin_numbers = int(maxDistance/binSize)
        CAduringStim = [[[] for _ in range(bin_numbers)] for _ in range(stimResults.shape[1])]
        activatedNeuronsDuringStim = np.zeros([block_No,bin_numbers])

        for iROI in range(stimResults.shape[0]):
            for iBlock in range(stimResults.shape[1]):
                binNo = math.floor((distanceFromElectrode[iROI]/maxDistance)/(1/bin_numbers))
                #if activatedNeurons[iROI][iBlock] == 1:
                CAduringStim[iBlock][binNo].append(np.mean(stimAvgs[iROI, iBlock, :]))
                if activatedNeurons[iROI][iBlock] == 1:
                    activatedNeuronsDuringStim[iBlock][binNo] += 1
                    #CAduringStim[iBlock][binNo].append(stimAvgs[iROI, iBlock, 0])


        distanceMeans = np.empty([stimResults.shape[1], bin_numbers])

        #plot distance vs. mean calcium activity and distance vs. activated neurons
        fig4, axs = plt.subplots(2,2)
        x_axis = ['0-50','50-100','100-150','150-200','200-250','250-300','300-350','350-400','400-450','450-500','500-550','550-600']
        for iBlock in range(stimResults.shape[1]):
            for iBin in range(bin_numbers):
                    distanceMeans[iBlock][iBin] = np.mean(CAduringStim[iBlock][iBin])

            axs[0,0].plot(x_axis,distanceMeans[iBlock, :])
            axs[1,0].plot(x_axis,activatedNeuronsDuringStim[iBlock,:])
        axs[0,0].legend(legend)
        axs[1,0].legend(legend)

        #calculate and plot mean distance of activated neurons vs. blocks
        distancesPerBlock = [[] for _ in range(block_No)]
        for iROI in range(ROI_No):
            for iBlock in range(block_No):
                if activatedNeurons[iROI][iBlock] == 1:
                    distancesPerBlock[iBlock].append(distanceFromElectrode[iROI])


        meanDistancesPerBlock = np.empty([block_No])
        print(meanDistancesPerBlock.shape)
        for iBlock in range(block_No):
            meanDistancesPerBlock[iBlock] = np.mean(distancesPerBlock[iBlock], axis = 0)
        axs[0,1].plot(legend, meanDistancesPerBlock)


        plt.show()

#scratch_1
def scratch_val(expDir):
    '''
    :param expDir:
    :return:
    '''
    base_dir = Path(expDir)
    filenames = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
    for dir in os.listdir(filenames):
        distances = np.load(expDir + '/' + dir + '/suite2p/plane0/distances.npy', allow_pickle=True)
        F0 = np.load(expDir + '/' + dir + '/suite2p/plane0/F0.npy', allow_pickle=True)
        #iscell = np.load(expDir + '/' + dir + '/suite2p/plane0/iscell.npy', allow_pickle=True)

        distanceValues = distances[:,2]
        plt.hist(distanceValues, bins=30, color='skyblue', edgecolor='black')
        plt.plot(F0[48,:])
        plt.show()

def plot_stim_traces(expDir, frame_rate=31, num_repeats=6, num_stims_per_repeat=5, list_of_file_nums=None):
    '''
    Plot calcium traces around stimulation timepoints in a grid
    
    Parameters
    ----------
    expDir : str
        Path to the experiment directory
    frame_rate : int
        Frame rate of the recording (default 31 Hz)
    num_repeats : int
        Number of times the stimulation sequence was repeated (default 6)
    num_stims_per_repeat : int
        Number of different stimulation amplitudes per repeat (default 5)
    list_of_file_nums : list of lists
        List of file numbers to analyze, e.g. [[11], [12]] for analyzing merged_11 and merged_12
    '''
    base_dir = Path(expDir)
    print(f"Looking for directories in: {base_dir}")
    
    # If list_of_file_nums is provided, only analyze those specific directories
    if list_of_file_nums is not None:
        dirs_to_analyze = []
        print(f"Looking for directories matching: {list_of_file_nums}")
        
        # First, print all available directories
        all_dirs = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
        print(f"Available directories: {all_dirs}")
        
        for sublist in list_of_file_nums:
            suffix = '_'.join(map(str, sublist))
            print(f"\nLooking for suffix: {suffix}")
            matching_dir = None
            
            # Original directory matching logic
            for file in base_dir.iterdir():
                if file.name.startswith('merged'):
                    print(f"Checking directory: {file.name}")
                    if file.name.endswith(suffix):
                        matching_dir = file.name
                        print(f"Found matching directory: {matching_dir}")
                        break
            
            if matching_dir:
                dirs_to_analyze.append(matching_dir)
            else:
                print(f"No matching directory found for suffix {suffix}")
    else:
        # If no list provided, analyze all merged directories
        dirs_to_analyze = [file.name for file in base_dir.iterdir() if file.name.startswith('merged')]
        print(f"Analyzing all directories: {dirs_to_analyze}")
    
    print(f"\nDirectories to analyze: {dirs_to_analyze}")
    
    for dir in dirs_to_analyze:
        print(f"\nAnalyzing directory: {dir}")
        # Load required data
        F_path = expDir + '/' + dir + '/suite2p/plane0/F.npy'
        stim_start_times_path = expDir + '/' + dir + '/stimTimes.npy'
        print(f"Loading data from: {F_path}")
        print(f"Loading stim times from: {stim_start_times_path}")
        
        F = np.load(F_path, allow_pickle=True)
        stim_start_times = np.load(stim_start_times_path, allow_pickle=True)
        
        # Rest of the function remains the same...



