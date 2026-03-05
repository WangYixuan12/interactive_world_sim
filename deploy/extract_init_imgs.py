"""Extract the first-frame init images from HDF5 episode files.

Run once from the repo root after downloading demo data:
    python deploy/extract_init_imgs.py
"""

from pathlib import Path

import cv2
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

deploy_dir = Path(__file__).parent.resolve()
data_dir = deploy_dir / "data"

TASKS = [
    {
        "hdf5": data_dir / "real_pusht_epi_0.hdf5",
        "obs_key": "camera_1_color",
        "out": deploy_dir / "real_pusht_topdown" / "init.png",
    },
    {
        "hdf5": data_dir / "real_bimanual_rope_epi_0.hdf5",
        "obs_key": "camera_0_color",
        "out": deploy_dir / "real_bimanual_rope_cam0" / "init.png",
    },
    {
        "hdf5": data_dir / "real_single_grasp_epi_0.hdf5",
        "obs_key": "camera_0_color",
        "out": deploy_dir / "real_single_grasp_cam0" / "init.png",
    },
    {
        "hdf5": data_dir / "real_bimanual_sweep_epi_0.hdf5",
        "obs_key": "camera_0_color",
        "out": deploy_dir / "real_bimanual_sweep_cam0" / "init.png",
    },
]

for task in TASKS:
    hdf5_path = task["hdf5"]
    if not hdf5_path.exists():
        print(f"Skipping {hdf5_path.name} (not found)")
        continue
    data, _ = load_dict_from_hdf5(str(hdf5_path))
    img = data["obs"]["images"][task["obs_key"]][0]
    img = center_crop(img, (128, 128))
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out_path = task["out"]
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path.relative_to(deploy_dir.parent)}")

print("Done.")
