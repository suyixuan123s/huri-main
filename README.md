# Deploy
## 0. RAPID:
Find rapid files in [yumi](drivers%2Fyumi) folder.
## 1. install the python 3.9.13 and CUDA 117 Phoxi-Control 1.75.0
- Compile the phoxi server following [phoxi_server.md](docs%2Fphoxi_server.md)
- Change the path for `PhoXiControl-1.7.5\API\bin` in the [__init__.py](drivers%2Fdevices%2Fphoxi%2F__init__.py)
## 2. Install the packages
```bash
pip install --no-index --find-links=<path> -r requirements.txt
```
- **Note1:** Change the `<path>` to the path where the packages are stored.
  
- **Note2:** It needs to manually install the `gym` package.

## 3. Calibrate the camera
1. Follow instructions in [calibration.md](docs%2Fcalibration.md)
   1. It generates a affine matrix file in the [huri/data/calibration](huri%2Fdata%2Fcalibration) directory.
2. Refine the calibration manually using [adjust_calibration_matrix.py](huri%2Fvision%2Fcalibration%2Fadjust_calibration_matrix.py)
    1. Change the `AFFINE_MAT_PATH` in the script to the calibration file you want to refine.
    2. Copy the output refined calibration matrix in the console and paste it in the original calibration file.
3. Change `SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH` in the [constants.py](huri%2Fcore%2Fconstants.py) to the refined calibration file.

## 4. Test vision system & Task planning
### 4.1 Test vision system
1. Determine the height of the table: [height_evaluation.py](huri%2Fcomponents%2Fdeploy%2Fvision%2Fheight_evaluation.py)
   - **Note:** It outputs the height value in the console.
2. Change `TABLE_HEIGHT` in the [constants.py](huri%2Fcomponents%2Fexe%2Fconstants.py) to the determined height.
2. Run [test_tube_reco.py](huri%2Fcomponents%2Fdeploy%2Fvision%2Ftest_tube_reco.py)

### 4.2 Test task planning
1. Run [task_solver_test.py](huri%2Fcomponents%2Ftask_planning%2Ftask_solver_test.py)

## 5. Run 
- Main program: [main_exe_v_0_0_3.py](huri%2Fcomponents%2Fexe%2Fversion%2Fmain_exe_v_0_0_3.py)
- GUI: [animation.py](huri%2Fcomponents%2Fexe%2Fversion%2Fanimation.py)

## 6. Capture data (If needed)
[capture_data.py](huri%2Fexamples%2Fvision%2Fcapture_data.py)