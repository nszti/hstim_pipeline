import matlab.engine

def run_cellreg_matlab(tiff_directory):
    '''

    Parameters
    ----------
    tiff_directory

    Returns
    -------
    Runs the cellreg registration process
    '''
    cellreg_dir = tiff_directory + 'cellreg_files'
    cellreg_dir_string = f"{cellreg_dir}"
    print(cellreg_dir_string)
    eng = matlab.engine.start_matlab()
    eng.cd(r'c:\Hyperstim\hstim_pipeline')
    eng.run_cellreg_w_python_api(cellreg_dir_string,1.07,'Non-rigid',1, nargout=0)
    eng.quit()

