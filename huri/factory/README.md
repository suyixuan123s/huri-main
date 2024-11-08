# Tutorial for Generating Pick and Place Motions for Test Tube Rearrangements

## Prerequisites
- **Update YUMI controller to the latest version**: Overwrite RAPID files in the YUMI controller using the `SERVER_L.mod` and `SERVERL_R.mod` in the folder [yumi](..%2F..%2Fdrivers%2Fyumi).
- **Use Trac-IK Solver to Improve IK Sovling Speed**:
    1. Compile the Trac-IK following the tutorial in the https://github.com/chenhaox/pytracik
    2. Copy the generated `pytracik.pyd` file to the [trac_ik](..%2F..%2Frobot_sim%2Frobots%2Fyumi%2Ftrac_ik)
3. Turn on the YUMI and start `SERVER_L` and `SERVER_R` in the YUMI controller.

## Generate Offline Pick-and-Place Motions
1. Generate IK solutions for the test tube rearrangement task using [tube_manipulation.py](tube_manipulation.py).
    The generated IK solutions will be saved as the `iks.pkl` in side the same folder.
2. Run the generated results using the [pick_and_place.py](pick_and_place.py)