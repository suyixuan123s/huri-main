# File Structure
- `component`: main functionality of the program
    - `control`
        - `yumi_con.py`: Yumi high-level control APIs
        - `yumi_remote_jog.py`: A remote jogging program for yumi
    - `debug`: programs for debug
    - `exe`: the folder contains program to real execution the test tube arrangement task 
        - `exe_logging.py`: define the logging sys
        - `executer.py`: the program to control yumi to execute the motions generated in simulation
        - `main_vision_feedback3.py`: the main program for the task
        - `utils.py`: utility functions
    - `gui`: TK Gui interface
    - `multiprocessing`: deprecated in the future
    - `pipeline`: define structures for data
    - `planning`: grasp planning and pick and place planning
    - `utils`: utility functions
    - `vision`
        - `extract.py`: extract point cloud
        - `tube_detector`: detect the test tube on the rack
- `core`: important utility functions
    - `common_import.py`: contains widely used module to optimize imports
    - `base_boost`: boost the Showbase (TODO: move it to component folder)
    - `file_sys`: work directories and functions to save data
    - `constants`: constants for sensor server ip address and calibration matrix
    - `panda3d_utils.py`: utility function for simulator (TODO: move it to component folder)
    - `print_tool`: some tool for better print (TODO: move it to component/utils folder)
    - `utils`: some utility functions
- `data`: the folder to save experiment data
- `definitions`: the data structure for rack and test tube
- `examples`: some examples of the system
- `learning`: code for machine learning
- `task_planning`: task planner for the test tube arrangement task
    - `task_puzzle_learning_solver.py`: the learning-based sequence planner
    - `task_puzzle_solover.py`: the A* based sequence planner
- `test`: some functionality under test
- `vision`: vision module (TODO: integrated with component/vision)
    - `calibration`: tools for point cloud calibration
    - `data`: saved experiment data
    - `template`: saved test tube and test tube rack point cloud template
    - `yolo`: yolo module (old)
    - `yolov6`: yolov5 6.0 module
    

    
    
    
work2: Cobot
work3: 
work4: nothing
work5: nothing