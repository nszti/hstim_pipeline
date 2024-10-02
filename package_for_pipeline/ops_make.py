import numpy as np
from suite2p import default_ops

ops = default_ops()

#Main settings
ops['nplanes'] = 1
ops['nchannels'] = 1
ops['tau'] = 0.62 #GCaMP6f
#ops['tau'] = 1.25 #GCaMP6s
ops['fs'] = 31
#Registration
ops['nimg_init'] = 200
ops['batch_size'] = 500
ops['maxregshift'] = 0.1
ops['reg_tif'] = True
#Nonrigid
ops['block_size'] = 128,128
ops['snr_threshold'] = 1.2
ops['maxregshiftNR'] = 5.0
#Functional detection
ops['sparse_mode'] = True
ops['denoise'] = True
ops['spatial_scale'] = 2
ops['connected'] = True  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
ops['threshold_scaling'] = 0.25 #GCaMP6f
#ops['threshold_scaling'] = 0.1  #GCaMP6s, 0.15-0.1
ops['max_overlap'] = 0.7
ops['max_iterations'] = 20
ops['high_pass'] = 300
ops['spatial_hp_detect'] = 25.0

np.save('C:/Hyperstim/pipeline_pending/.suite2p/ops/ops_GCaMP6f.npy', ops) #GCaMP6f
#np.save('C:/Hyperstim/pipeline_pending/.suite2p/ops/ops_GCaMP6s.npy', ops) #GCaMP6s