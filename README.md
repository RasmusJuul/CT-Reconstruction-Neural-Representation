Code is work in progress, not user friendly yet, preparation of data is not streamlined.

**Install**
* run `pip install -r requirements.txt`
* run `pip install -e .`
  
**Data**
* Add volumetric data to the data folder, can be in stored in individual files or together in a hdf5 dataset
```
├── data
│   ├── your_dataset
│   │   ├──  Train
│   │   │   ├── vol1.tif
│   │   │   ├── vol2.tif
│   │   │   ├── vol3.tif
│   │   │   ├── All_volumes.hdf5
│   │   ├──  Test
│   │   │   ├── vol4.tif
```
* Run ``
* Run `create_ray_data.py`

**Train model/Reconstruct volume**
* Run `train_raygan.py --experiment-name xxx --data-path your_dataset/some_name_youll_have_from_previous_step --activation-function relu --num-points 512 --model-lr 1e-3 --d-lr 1e-4 --num-epochs 1000 --batch-size 3500 --num-workers 16 --encoder hashgrid --adversarial-weight 1e-3 --ray-data-path $MYTMP/combined_interpolated_points_16.hdf5 --midpoint 0.15 --sharpness 0.05 --extra-positions-path your_dataset/200proj_positions.npy --extra-ray-data-path $MYTMP/combined_interpolated_points_200.hdf5 --extra-batch-size 1500 --smoothness-weight 5e-5 --curvature-weight 0 --consistency-weight 5e-2 --fml-weight 0 --dilated --multiscale`

