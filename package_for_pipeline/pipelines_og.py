import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
from package_for_pipeline.hdf5io.hdf5io import Hdf5io
from cv2 import bitwise_not
#from caiman_analysis import run_caiman_on_tiff
from mesc_loader import extract_useful_xml_params
from general import ascii_to_str, find_frame_index_from_timestamp


def analyse_mesc_file(filepath, plot_curves=False, print_all_attributes=False, stim_trig_channel="DI3"):
    with Hdf5io(filepath) as mesc_file:
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

        f = open("C:/Hyperstim/pipeline_pending/mesc_preprocess/trigger.txt", "w")
        f2 = open("C:/Hyperstim/pipeline_pending/mesc_preprocess/fileID.txt", "w")
        f3 = open("C:/Hyperstim/pipeline_pending/mesc_preprocess/frameNo.txt", "w")
        for unit_id in units:
        #for unit_id in range(len(units)):
            selected_unit = selected_session[unit_id]
            #selected_unit = selected_session['MUnit_' + str(unit_id)]
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
                            print("Has X axis data.")
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
                                    print(i)
                                    stim_start_indices.append(i)
                            print("Stimulation start indices (ms):", stim_start_indices)


            # ---Printing Unit attributes---
            if print_all_attributes:
                for attr_key in selected_unit._v_attrs._v_attrnames:
                    try:
                        print(attr_key, ascii_to_str(selected_unit._v_attrs[attr_key]) or None)
                    except TypeError:
                        print(attr_key, ' has caused type error.')
                    except OverflowError:
                        print(attr_key, ' has caused overflow error.')

            frame_time_ms = selected_unit._v_attrs['ZAxisConversionConversionLinearScale'] + selected_unit._v_attrs['ZAxisConversionConversionLinearOffset']

            # ---Extracting data from Unit XML parameters---
            params = extract_useful_xml_params(ascii_to_str(selected_unit._f_getattr('MeasurementParamsXML')))
            params['framerate'] = 1 / (frame_time_ms / 1000)
            pprint(params)

            # ---Load recording in Unit---
            # Load and invert image array (in all test files the recording was in Channnel_0 and there were no other channels)
            image_seq = bitwise_not(selected_unit['Channel_0'][()])  # load & invert image array
            image_seq.squeeze()
            print(image_seq.shape, image_seq.dtype)

            frame_timestamps = np.arange(0, frame_time_ms * image_seq.shape[0], frame_time_ms)

            try:
                if len(stim_start_indices) > 0:
                    stim_start_frame_indices = [find_frame_index_from_timestamp(timestamp, frame_timestamps) for timestamp in stim_start_indices]
                    f.write(str(stim_start_frame_indices[0]))
                    f.write("\n")
                    f2.write(unit_id)
                    f2.write("\n")
                    f3.write(str(len(image_seq)))
                    f3.write("\n")
                    print("Stimulation start 2p frame indices:", stim_start_frame_indices)
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
        print(len(image_seq))

if __name__ == '__main__':
    folder = Path("C:/Hyperstim/pipeline_pending/mesc_preprocess/")
    filename = "2024_02_09_in_vivo_GCAMP6F_2.mesc"
    analyse_mesc_file(folder/filename, print_all_attributes=True, plot_curves = True)
