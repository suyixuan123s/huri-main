import huri.core.file_sys as fs

command = ".\ExternalCameraExample_Release.exe --calibrate D:\\chen\\phoxi_server_tst\\calib_external_cam_custom\\Data\\1.praw"
image_save_path = fs.Path(__file__).parent / "data"
for file_path in image_save_path.glob("**/*"):
    if not file_path.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        continue
    command += f" {str(file_path)}"
print(command)