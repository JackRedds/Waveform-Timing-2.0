from pathlib import Path

work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
psp_data_dir = work_dir / "psp_data"
output_data_dir = work_dir / "output_data"
wave_finder_data_dir = output_data_dir / "Wave-Train-Finder"
wave_train_data_dir = output_data_dir / "Wave-Train"
solitary_wave_data_dir = output_data_dir / "Solitary-Wave"
plot_dir = work_dir / "plot"

for dir in [psp_data_dir, wave_finder_data_dir, wave_train_data_dir, solitary_wave_data_dir, plot_dir]:
    dir.mkdir(parents=True, exist_ok=True)
