from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xmltodict
from cv2 import bitwise_not
from general import ascii_to_str

# folder_path = Path("/media/hillierlab/Data/2P/Test_data")
folder_path = Path("/media/hillierlab/Data/2P/Rat/Rat 110/2023_07_13 2P")
filename = "2023_07_13_GRIN_rat_zstack.mesc"
# MUnit = '2'
plot_curves = False


def extract_useful_xml_params(measurement_params_xml):
    measurement_params = xmltodict.parse(measurement_params_xml)
    # TODO needed params for caiman: fps, signal_decay_time, half_size_of_neuron_in_pixels

    useful_params = {"pockels_cell": float(measurement_params['Task']['Devices']['Gear'][0]['param'][4]['@value']),
                     "pmt_ug": float(measurement_params['Task']['Devices']['Gear'][1]['param'][1]['@value']),
                     "scan_change_waveplate": float(measurement_params['Task']['Devices']['APTMotor'][0]['param'][0]['@value'])}
    for i in range(len(measurement_params['Task']['Params']['param'])):
        try:
            useful_params[measurement_params['Task']['Params']['param'][i]['@name']] = float(measurement_params['Task']['Params']['param'][i]['@value'])
        except ValueError:
            useful_params[measurement_params['Task']['Params']['param'][i]['@name']] = measurement_params['Task']['Params']['param'][i]['@value']

    return useful_params

    """
    fig, ax = plt.subplots()
    frame = 0
    im = plt.imshow(image_seq[frame], cmap='gray')
    plt.colorbar()

    def update(*args):
        global frame

        im.set_array(image_seq[frame])

        frame += 1
        frame %= len(image_seq)

        return im,

    ani = animation.FuncAnimation(fig, update, interval=100)
    plt.show()
    """
