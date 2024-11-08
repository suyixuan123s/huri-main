# Preparation

The preparation step only needs to be executed once.

1. Update `SENSOR_INFO` in [constants.py](..%2Fcore%2Fconstants.py)
   ```python
   class SENSOR_INFO:
      IP_ADR_DEPTH_SENSOR
      PNT_CLD_CALIBR_MAT_PATH
   ```
    - `IP_ADR_DEPTH_SENSOR`: IP address for the Phoxi
      server ([phoxi_server.py](..%2F..%2Fdrivers%2Frpc%2Fphoxi%2Fphoxi_server.py))
    - `PNT_CLD_CALIBR_MAT_PATH`: Calibration matrix for the point cloud (This matrix transforms the point cloud from
      camera coordinates to the simulation world coordinates).

2. Generate observation poses
    - Run [generate_in_hnd_annotation_iks.py](generate_in_hnd_annotation_iks.py)

---   

# Data Collection for Tube

1. Start the YUMI robot and the Phoxi server:
    - Power on the robot and run the SERVER rapid program.
    - Run the Phoxi Server ([phoxi_server.py](..%2F..%2Fdrivers%2Frpc%2Fphoxi%2Fphoxi_server.py)).
2. Configure parameters for automated data collection ([constant.py](constant.py)).
    - `TUBE_NAME`: The name of the test tube.
    - `TUBE_ID`: The index of the test tube.
    - `WORK_ARM`: The arm used to hold the test tube.
    - `SAVE_PATH`: [Important] The path to save the collected data.

3. Run [in_hand_annotation.py](in_hand_annotation.py)
    - If `WORK_ARM` is set to `both`, the robot will perform as shown below:
    - ![1.jpg](imgs%2F1.jpg)
    - Instruct the robot to grasp the test tube:
        - Horizontally grasp the test tube.
        - Ensure a secure grip on the test tube.
        - **[Note] The YUMI gripper might release due to overheating.** It's advisable to use tape to secure the test
          tube in the hand.
    - The interface appears as follows:
        - ![2.png](imgs%2F2.png)
            - Click the `Get data and render` button to display the 2D image and point cloud data.
              ![3.png](imgs%2F3.png)

4. Adjust the size of the masks to encompass the point clouds of the caps while excluding excess noise:
    - **Ensure the mask isn't too large to prevent excessive noise.**
    - ![4.png](imgs%2F4.png)
    - An example of an adjusted mask for the caps' point clouds. The annotated results are also displayed in the 2D
      image.
    - ![5.png](imgs%2F5.png)

5. Click the `Move robot to next position` button and then the `Get data and render` button to confirm the correctness
   of the annotation results in the 2D image:
    - ![6.png](imgs%2F6.png)

6. Click the `Auto Collect Data` button to initiate automated data collection. Wait until the process is complete.
    - Files will be saved as `.pkl` format.

---

# Data Collection for Rack

1. Locate the variable `SAVE_PATH` in [on_table_annotation.py](on_table_annotation.py).
    - Update its value to specify the desired directory where you want to save the collected data
2. Run [on_table_annotation.py](on_table_annotation.py). An interface appears as follows:
   ![7.png](imgs%2F7.png)
3. Click `Get data and render` button to visualize the 2D image and point cloud data:
   ![8.png](imgs%2F8.png)
   4Adjust the mask sizes to encompass the point clouds of the rack:
   ![9.png](imgs%2F9.png)
5. After adjusting the masks, click the `Get data and render and save` button to save the annotation results.
6. Change the position of the rack and repeat the annotation process for different positions.

---

# Postprocessing

1. Output the annotation results and images using [postprocessing.py](postprocessing.py)
    - Open [postprocessing.py](postprocessing.py)
    - Locate the variable `save_data_path` and modify it to specify the desired path for saving the images and labels.
      Update it to your preferred directory.

---

# Data Synthesis

1. Download the BG2K background dataset:
    - Go to the following link: https://drive.google.com/drive/folders/1ZBaMJxZtUNHIuGj8D8v3B9Adn8dbHwSS
    - Download the `train` directory from the provided Google Drive folder.
2. Set the BG2K dataset path in [copy_paste.py](copy_paste.py).
    - Locate the variable `BG2K_PATH` and modify its value to the path where you downloaded and stored the BG2K dataset.
3. Configure the destination path for saving images and labels in [copy_paste.py](copy_paste.py):
    - Locate the variable `save_data_path` in [copy_paste.py](copy_paste.py).
    - Update its value to specify the desired directory where you want to save the synthesized images and associated
      labels.
4. Update the following code section in [copy_paste.py](copy_paste.py) based on your collected data:
   ```python
    copy_paste_annotation(cp_paths_and_classes=[[r'E:/huri_shared/huri/data/data_annotation/EXP/WHITE_CAP', 1], ],
                          bg_paths_and_classes=[[r'D:\chen\huri_shared\huri\data\data_annotation\EXP\RACK', 0]],
                          num_imgs=10,
                          save_path=save_data_path,
                          num_tubes=25,
                          toggle_debug=False,
                          random_bg=True, )
    ```
    - `cp_paths_and_classes`: A list of lists containing `[path_annotation, class_id]` pairs for your collected data.
        - `path_annotation1`: The path where the `.pkl` files generated by the automated data collection are stored.
        - `class_id1`: The class index of the data.
    - `bg_paths_and_classes`: A list containing the path to the rack images and their corresponding class index.
    - `num_imgs`: A list containing the path to the rack images and their corresponding class index.
    - `num_tubes`: The maximum number of tubes to include in the synthesized images.
    - `random_bg`: Choose whether to randomize the background of the synthesized in-rack test tubes.
    - `toggle_debug`: Set to `True` if you are debugging.