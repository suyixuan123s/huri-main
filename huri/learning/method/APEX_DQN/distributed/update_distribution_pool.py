""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""
import shutil
import huri.core.file_sys as fs


def update(start, goal):
    # Copy all files from the 'start' directory to the 'goal' directory
    for file_path in start.iterdir():
        if file_path.is_file():
            shutil.copy(file_path, goal)

def update_restore():
    start = fs.Path(r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed')
    goal = fs.Path(r"E:\learning\huri\learning\method\APEX_DQN\distributed")

    update(start, goal)

    start = fs.Path(r'E:\huri_shared\huri\learning\env\rack_v3')
    goal = fs.Path(r"E:\learning\huri\learning\env\rack_v3")

    update(start, goal)

    start = fs.Path(r'E:\huri_shared\huri\learning\A_start_teacher')
    goal = fs.Path(r"E:\learning\huri\learning\A_start_teacher")

    update(start, goal)

    print("All files have been copied successfully.")

if __name__ == '__main__':
    update_restore()
