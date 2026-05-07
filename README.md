**Processing MRI**
- pipeline_01_process_mri.py

**Running with OPM data**
- pipeline_08_opm_data_convert.py
- inversion_s00_opm.ipynb
- inversion_s01_opm.ipynb

**Running with magnetometer-only data (original shared datafiles)**
- pipeline_08_mag_sensor_data_spm_convert.py
- get_fiducial_coords_mag_sensors.py
- inversion_s01_squid_mag_sensors.ipynb
- inversion_s03_squid_mag_sensors.ipynb

**Running with all-sensor data (requires laMEG v0.1.4)**
- pipeline_02_raw_prep.py
- pipeline_03_pp_noise.py
- pipeline_04_ica_inspection.py
- pipeline_05_epoch.py
- pipeline_06_epoch_qc.py
- pipeline_07_autoreject.py
- pipeline_08_all_sensor_data_spm_convert.py
- get_fiducial_coords_all_sensors.py
- inversion_s01_squid_all_sensors.ipynb
- inversion_s03_squid_all_sensors.ipynb
