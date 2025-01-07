import numpy as np
import pandas as pd
from pytplot import get_data as get_psp_data
import pyspedas
from wave_timing.utils import directory
import matplotlib.pyplot as plt


class get_data:

    def __init__(self, start_time: str, end_time: str):
        self.start_time = start_time
        self.end_time = end_time


    def get_vac_data(self):
        data_type = 'dfb_dbm_vac'

        vac_data = pyspedas.psp.fields(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l2',
            time_clip=True,
            file_path=directory.psp_data_dir
            )

        vac1 = get_psp_data('psp_fld_l2_dfb_dbm_vac1').y
        vac2 = get_psp_data('psp_fld_l2_dfb_dbm_vac2').y
        vac3 = get_psp_data('psp_fld_l2_dfb_dbm_vac3').y
        vac4 = get_psp_data('psp_fld_l2_dfb_dbm_vac4').y
        vac5 = get_psp_data('psp_fld_l2_dfb_dbm_vac5').y
        time_TT2000 = get_psp_data('psp_fld_l2_dfb_dbm_vac_time_series_TT2000').times
        time = get_psp_data('psp_fld_l2_dfb_dbm_vac1').v[0]
        start_date = pd.to_datetime(time_TT2000, unit='s')

        dv1 = vac1 - (vac3 + vac4) / 2
        dv2 = (vac3 + vac4) / 2 - vac2
        dv3 = (vac1 + vac2) / 2 - vac3
        dv4 = vac4 - (vac1 + vac2) / 2
        dvsc = (vac1 + vac2 + vac3 + vac4) / 4
        dv5 = dvsc - vac5

        dv1 = pd.DataFrame(dv1.T, columns=start_date, index=time)
        dv2 = pd.DataFrame(dv2.T, columns=start_date, index=time)
        dv3 = pd.DataFrame(dv3.T, columns=start_date, index=time)
        dv4 = pd.DataFrame(dv4.T, columns=start_date, index=time)
        dv5 = pd.DataFrame(dv5.T, columns=start_date, index=time)

        return dv1, dv2, dv3, dv4, dv5

    def get_vdc_data(self):
        data_type = 'dfb_dbm_vdc'

        vdc_data = pyspedas.psp.fields(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l2',
            time_clip=True,
            file_path=directory.psp_data_dir
            )

        vdc1 = get_psp_data('psp_fld_l2_dfb_dbm_vdc1').y
        vdc2 = get_psp_data('psp_fld_l2_dfb_dbm_vdc2').y
        vdc3 = get_psp_data('psp_fld_l2_dfb_dbm_vdc3').y
        vdc4 = get_psp_data('psp_fld_l2_dfb_dbm_vdc4').y
        vdc5 = get_psp_data('psp_fld_l2_dfb_dbm_vdc5').y
        time_TT2000 = get_psp_data('psp_fld_l2_dfb_dbm_vdc_time_series_TT2000').times
        time = get_psp_data('psp_fld_l2_dfb_dbm_vdc1').v[0]
        start_date = pd.to_datetime(time_TT2000, unit='s')

        dv1 = vdc1 - (vdc3 + vdc4) / 2
        dv2 = (vdc3 + vdc4) / 2 - vdc2
        dv3 = (vdc1 + vdc2) / 2 - vdc3
        dv4 = vdc4 - (vdc1 + vdc2) / 2
        dvsc = (vdc1 + vdc2 + vdc3 + vdc4) / 4
        dv5 = dvsc - vdc5

        dv1 = pd.DataFrame(dv1.T, columns=start_date, index=time)
        dv2 = pd.DataFrame(dv2.T, columns=start_date, index=time)
        dv3 = pd.DataFrame(dv3.T, columns=start_date, index=time)
        dv4 = pd.DataFrame(dv4.T, columns=start_date, index=time)
        dv5 = pd.DataFrame(dv5.T, columns=start_date, index=time)

        return dv1, dv2, dv3, dv4, dv5




    def get_mag_data(self):
        data_type = 'mag_SC_4_Sa_per_Cyc'

        data = pyspedas.psp.fields(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l2',
            time_clip=True,
            file_path=directory.psp_data_dir
            )

        mag_data = get_psp_data('psp_fld_l2_mag_SC_4_Sa_per_Cyc')
        date = pd.to_datetime(mag_data.times, unit='s')
        mag_comp = pd.DataFrame(mag_data.y, columns=['Bx', 'By', 'Bz'], index=date)
        B = np.sqrt(mag_comp.Bx ** 2 + mag_comp.By ** 2 + mag_comp.Bz ** 2)

        mag_comp.insert(3, '|B|', B, True)

        return mag_comp


    def get_sw_data(self):
        data_type = 'sf00_l3_mom'

        data = pyspedas.psp.spi(
            trange=[self.start_time, self.end_time],
            datatype=data_type,
            level='l3',
            time_clip=True,
            file_path=directory.psp_data_dir
        )

        sw_data = get_psp_data('psp_spi_VEL_SC')
        date = pd.to_datetime(sw_data.times, unit='s')
        sw_comp = pd.DataFrame(sw_data.y, columns=['Vx', 'Vy', 'Vz'], index=date)
        V = np.sqrt(sw_comp.Vx ** 2 + sw_comp.Vy ** 2 + sw_comp.Vz ** 2)

        sw_comp.insert(3, '|V|', V, True)

        return sw_comp

def save_data(data: pd.DataFrame, file_path, file_name: str):
    data.to_csv(file_path / file_name)
    print(f"Data File {file_name} Saved")
