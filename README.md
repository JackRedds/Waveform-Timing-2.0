# Waveform Timing 2.0
Updated Timing code

## TODO
### Jack:
- [x] write pyproject.toml
- [x] write .gitignore file
- [x] write enviroment.yml
- [ ] write test codes
- [ ] look into assert statements
- [ ] plan out scripts
- [ ] write leftover functions
- [x] write Makefile
- [x] fork pyspedas repo so data loc can be changed
- [ ] Look at wave train finder code

### Petra
- [ ] find_nearest
- [ ] angle_wrt_v1
- [ ] sample_rate
- [ ] rotate_mag_fld
- [ ] FirBandPass
- [ ] filter_fir

## Code Set-Up:
To set up virtual enviroment and download wave-timing code along with dependencies simply run `$make install` if you have make installed on your computer if not run:
```
conda env create -f enviroment.yml
conda run -n wave-timing poetry install
```
