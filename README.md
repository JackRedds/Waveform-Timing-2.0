# Waveform Timing 2.0
Updated Timing code

## To Do
### Jack:
- [x] write pyproject.toml
- [ ] **write .gitignore file**
- [x] write enviroment.yml
- [ ] write test codes
- [ ] look into assert statements
- [ ] plan out scripts
- [ ] write leftover functions
- [x] write Makefile

### Petra
- [ ] find_nearest
- [ ] angle_wrt_v1
- [ ] sample_rate
- [ ] rotate_mag_fld
- [ ] FirBandPass
- [ ] filter_fir

## Code Set-Up:
To set up virtual enviroment and download wave-timing code along with dependencies simply run '$make install' if you have make installed on your computer if not run:
```
conda env create -f enviroment.yml
conda run -n wave-timing poetry install
```
