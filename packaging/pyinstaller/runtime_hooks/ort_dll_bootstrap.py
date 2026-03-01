import ctypes
import os
import sys

_HANDLES = []


def configure_ort_dll_search_path():
    if sys.platform != "win32" or not hasattr(sys, "_MEIPASS"):
        return

    base_dir = sys._MEIPASS
    search_dirs = [os.path.join(base_dir, "onnxruntime", "capi"), base_dir]

    if hasattr(os, "add_dll_directory"):
        for directory in search_dirs:
            if os.path.isdir(directory):
                _HANDLES.append(os.add_dll_directory(directory))

    for dll_name in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
        for directory in search_dirs:
            dll_path = os.path.join(directory, dll_name)
            if os.path.isfile(dll_path):
                ctypes.WinDLL(dll_path)
                break


configure_ort_dll_search_path()
