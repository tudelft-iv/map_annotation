#!/bin/sh
# this file is meant to be sourced by bash to set environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	echo "This file is meant to be sourced. Please run"
	echo "source $0"
	exit 1
fi

# set variable to indicate that this was sourced
export MA_SOURCED=

# absolute path of this script file
export MA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
export MA_DATA_DIR="${MA_DIR}/data"
export MA_SRC_DIR="${MA_DIR}/src"

export PYTHONPATH="$MA_DIR:$PYTHONPATH"
export PYTHONPATH="$MA_SRC_DIR:$PYTHONPATH"

conda activate map-annotation

echo "Done"
