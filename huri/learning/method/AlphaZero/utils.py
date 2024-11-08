from typing import List, Tuple, Mapping, Union, Text, Any
import torch
import time
import sys
import os
import signal
import shutil


def load_checkpoint(ckpt_file: str, device: torch.device) -> Mapping[Text, Any]:
    return torch.load(ckpt_file, map_location=torch.device(device))


def create_checkpoint(state_to_save: Mapping[Text, Any], ckpt_file: str) -> None:
    torch.save(state_to_save, ckpt_file)


def get_time_stamp(as_file_name: bool = False) -> str:
    t = time.localtime()
    if as_file_name:
        timestamp = time.strftime('%Y%m%d_%H%M%S', t)
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return timestamp


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        print(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

        # # Listen to signals to exit process.

    if sys.platform.startswith('linux'):
        signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)


def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False


def delete_all_files_in_directory(folder_path):
    folder_path = str(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def is_folder_empty(folder_path):
    str(folder_path)
    return not os.listdir(folder_path)