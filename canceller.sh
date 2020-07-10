#!/bin/bash

KEYWORD=$1

echo "About to cancel the following jobs:\n"

squeue -u "lorell" | grep "$KEYWORD"

read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

squeue -u "lorell" | grep "$KEYWORD" | awk '{print $1}' | xargs -n 1 scancel
