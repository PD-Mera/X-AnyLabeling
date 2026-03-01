import os
import glob
import shutil
import subprocess
from PyQt6 import QtCore


def compile_resources(output: str, qrc: str) -> None:
    """Compile a .qrc file to a PyQt6-compatible resources.py."""
    if shutil.which("pyside6-rcc"):
        result = subprocess.run(
            ["pyside6-rcc", "-o", output, qrc],
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            with open(output, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("from PySide6", "from PyQt6")
            with open(output, "w", encoding="utf-8") as f:
                f.write(content)
            return
    if shutil.which("pyrcc5"):
        subprocess.run(["pyrcc5", "-o", output, qrc], check=True)
        with open(output, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("from PyQt5", "from PyQt6")
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        return
    print(
        "Error: neither pyside6-rcc nor pyrcc5 found. Install pyside6 or pyqt5-tools."
    )


supported_languages = ["en_US", "zh_CN"]
translations_path = "anylabeling/resources/translations"

for language in supported_languages:
    # Scan all .py files in the project directory and its subdirectories
    py_files = glob.glob(os.path.join("**", "*.py"), recursive=True)

    # Create a QTranslator object to generate the .ts file
    translator = QtCore.QTranslator()

    # Translate all .ui files into .py files
    ui_files = glob.glob(os.path.join("**", "*.ui"), recursive=True)
    for ui_file in ui_files:
        py_file = os.path.splitext(ui_file)[0] + "_ui.py"
        command = f"pyuic6 -x {ui_file} -o {py_file}"
        os.system(command)

    # Extract translations from the .py file
    command = f"pylupdate6 --no-obsolete {' '.join(py_files)} -ts {translations_path}/{language}.ts"
    os.system(command)

    # Compile the .ts file into a .qm file
    command = f"lrelease {translations_path}/{language}.ts"
    os.system(command)

compile_resources(
    output="anylabeling/resources/resources.py",
    qrc="anylabeling/resources/resources.qrc",
)
