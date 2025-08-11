import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from pathlib import Path
#import hdf5io
#from hdf5io.hdf5io import Hdf5io
from cv2 import bitwise_not
from mesc_loader import extract_useful_xml_params
from general import ascii_to_str, find_frame_index_from_timestamp

#NB!: Check trigger channel in MESc, modify if needed!!!!
def analyse_mesc_file(filepath, output_path,  plot_curves=False, print_all_attributes=False, stim_trig_channel="DI2"):
    '''

    Parameters
    ----------
    filepath: path to the MESc file
    output_path: output path, usually the directory which contains the MESc file
    plot_curves: wheather to plot curves in MUnit
    print_all_attributes:unit attributes
    stim_trig_channel: the channel from which the triggers have been initiated in MESc

    Returns
    -------
    fileID.txt: saves the tiff file recording numbers/IDs from MESc
    frameNo.txt: saves the frame number of each file
    trigger.txt: saves the start of the stimulation's frame number
    mesc_data.npy: contains all three datasets' dataframe into a .npy format
    '''
    dir= Path(output_path)
    file_to_search = f"fileID.txt"
    stim_start = []
    if file_to_search not in os.listdir(dir): #search for number of tiff files
        print(f"Extracting TIFF files")

        outer_directory = os.path.join(output_path, 'merged_tiffs')
        os.makedirs(outer_directory, exist_ok=True)

        triggers = []
        files_ids = []
        frame_nos = []
        mesc_filepath = f'{filepath}.mesc'
        with Hdf5io(mesc_filepath) as mesc_file:
            print(filepath)
            sessions = list(mesc_file.h5f.root._v_children.keys())
            if len(sessions) == 1:
                print("There is only a single MSession saved in this file. Selecting it.")
            else:
                print("Handling MESc file with multiple MSessions is not supported yet. Using the first one.")
            session_key = sessions[0]
            hdf_root = mesc_file.h5f.root
            selected_session = hdf_root[session_key]
            units = list(selected_session._v_children.keys())
            print("Number of Units is Session: ", len(units))

            f = open(f"{output_path}/trigger.txt", "w")
            f2 = open(f"{output_path}/fileID.txt", "w")
            f3 = open(f"{output_path}/frameNo.txt", "w")

            for unit_id in units:
                selected_unit = selected_session[unit_id]
                # Printing comment of MUnit
                comment = selected_unit._f_getattr('Comment')
                # Decoding ascii to string
                comment_str = ascii_to_str(comment)
                print(f"Unit {unit_id} comment: ", comment_str or "Empty")
                # ---Plotting curves in MUnit---
                if plot_curves:
                    for key in selected_unit._v_children.keys():
                        print(key)
                        if 'Curve' in key:
                            if 'CurveDataXRawData' in selected_unit[key]._v_children.keys():
                                plt.plot(selected_unit[key]['CurveDataXRawData'][()],
                                         selected_unit[key]['CurveDataYRawData'][()])
                                for attr_key in selected_unit[key]._v_attrs._v_attrnames:
                                    print(attr_key, ' ', ascii_to_str(selected_unit[key]._v_attrs[attr_key]))
                            else:
                                plt.plot(selected_unit[key]['CurveDataYRawData'][()])
                            plt.title(ascii_to_str(selected_unit[key]._v_attrs['Name']) + ' ' + key)
                            plt.xlabel(ascii_to_str(selected_unit[key]._v_attrs['CurveDataXConversionTitle']) + ' [' +
                                       ascii_to_str(selected_unit[key]._v_attrs['CurveDataXConversionUnitName']) + ']')
                            plt.ylabel(ascii_to_str(selected_unit[key]._v_attrs['CurveDataYConversionTitle']) + ' [' +
                                       ascii_to_str(selected_unit[key]._v_attrs['CurveDataYConversionUnitName']) + ']')
                            plt.show(block=False)

                            # ---Extracting stimulation start timepoints---
                            if stim_trig_channel in ascii_to_str(selected_unit[key]._v_attrs['Name']):
                                y_data = selected_unit[key]['CurveDataYRawData'][()]
                                stim_start_indices = []
                                for i in range(1, y_data.shape[0]):
                                    if y_data[i] > y_data[i-1]:
                                        stim_start_indices.append(int(np.round(i/20))) # after 2025.04.07
                                        stim_start.append(i)
                                #print(f"Stimulation start indices (ms): {stim_start_indices}")

                # ---Printing Unit attributes---
                '''if print_all_attributes:
                    print("belep")
                    for attr_key in selected_unit._v_attrs._v_attrnames:
                        print(attr_key, selected_unit._v_attrs[attr_key])
                        try:
                            print(attr_key, ascii_to_str(selected_unit._v_attrs[attr_key]) or None)
                        except TypeError:
                            print(attr_key, ' has caused type error.')
                        except OverflowError:
                            print(attr_key, ' has caused overflow error.')'''

                frame_time_ms = selected_unit._v_attrs['ZAxisConversionConversionLinearScale'] + selected_unit._v_attrs['ZAxisConversionConversionLinearOffset']

                # ---Extracting data from Unit XML parameters---
                params = extract_useful_xml_params(ascii_to_str(selected_unit._f_getattr('MeasurementParamsXML')))
                params['framerate'] = 1 / (frame_time_ms / 1000)
                pprint(params)
                #print(params['framerate'])

                # ---Load recording in Unit---
                # Load and invert image array (in all test files the recording was in Channnel_0 and there were no other channels)
                image_seq = bitwise_not(selected_unit['Channel_0'][()])  # load & invert image array
                image_seq.squeeze()
                #print(image_seq.shape, image_seq.dtype)

                frame_timestamps = np.arange(0, frame_time_ms * image_seq.shape[0], frame_time_ms)
                #print(image_seq.shape)
                try:
                    if len(stim_start_indices) > 0:
                        stim_start_frame_indices = [find_frame_index_from_timestamp(timestamp, frame_timestamps) for timestamp in stim_start_indices]
                        #print(stim_start_frame_indices)
                        triggers.append(stim_start_frame_indices[0])
                        files_ids.append(unit_id)
                        frame_nos.append(len(image_seq))

                        f.write(str(stim_start_frame_indices[0]))
                        f.write("\n")
                        f2.write(unit_id)
                        f2.write("\n")
                        f3.write(str(len(image_seq)))
                        f3.write("\n")
                        #print(f"Stimulation start 2p frame indices:{stim_start_frame_indices}")

                except UnboundLocalError:
                   pass

                # flipping only for matplotlib imshow
                image_seq = np.flip(image_seq, axis=1)

                if len(image_seq.shape) == 2:  # it's a single image, only showing it and saving as png
                    plt.imshow(image_seq, cmap='gray')
                    plt.show()
                else:
                    # Tiff export and caiman analysis
                    import tifffile
                    tifffile.imwrite(f"{mesc_file.filename[:-5]}_{unit_id}.tif", data=image_seq, shape=image_seq.shape)
                    # run_caiman_on_tiff(filepath=f"{mesc_file.filename[:-5]}_{unit_id}.tif", framerate=params['framerate'],
                    #                    decay_time=4, xy_resolution=(params['PixelSizeX'], params['PixelSizeY']))

            f.close()
            f2.close()
            f3.close()
            #print(len(image_seq))
            data = {'FileID' : files_ids, 'FrameNo': frame_nos, 'Trigger' : triggers}
            df = pd.DataFrame(data)
            output_dir_path = f'{output_path}merged_tiffs/mesc_data.npy'
            np.save(output_dir_path, df.to_numpy())
            #np.savetxt(output_path, output_dir_path, fmt='%12.8f %12.8f')
    print(f"Stimulation start indices (ms): {stim_start}")

