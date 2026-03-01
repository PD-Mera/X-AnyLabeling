import os
import shutil
import subprocess


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

for language in supported_languages:
    command = f"lrelease anylabeling/resources/translations/{language}.ts"
    os.system(command)

compile_resources(
    output="anylabeling/resources/resources.py",
    qrc="anylabeling/resources/resources.qrc",
)
