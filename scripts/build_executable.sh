#!/bin/bash
system=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SPEC_DIR="${ROOT_DIR}/packaging/pyinstaller/specs"
export X_ANYLABELING_ROOT="${ROOT_DIR}"

if [ "$system" = "win-cpu" ]; then
    echo "Building Windows CPU version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-win-cpu.spec"
elif [ "$system" = "win-gpu" ];then
    echo "Building Windows GPU version..."
    export X_ANYLABELING_DEVICE=GPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-win-gpu.spec"
elif [ "$system" = "linux-cpu" ];then
    echo "Building Linux CPU version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-linux-cpu.spec"
elif [ "$system" = "linux-gpu" ];then
    echo "Building Linux GPU version..."
    export X_ANYLABELING_DEVICE=GPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-linux-gpu.spec"
elif [ "$system" = "macos" ];then
    echo "Building macOS version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm "${SPEC_DIR}/x-anylabeling-macos.spec"
else
    echo "System value '$system' is not recognized."
fi
