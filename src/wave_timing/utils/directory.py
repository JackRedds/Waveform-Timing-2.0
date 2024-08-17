from pathlib import Path

work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
psp_data_dir = work_dir / "psp_data"
output_data_dir = work_dir / "output_data"
plot_dir = work_dir / "plot"

for dir in [psp_data_dir, output_data_dir, plot_dir]:
    dir.mkdir(parents=True, exist_ok=True)
