#!/bin/bash

echo "Syncing down from Slurm. Dry run..."

rsync -avhzrn lorell@slurm.ttic.edu:~/SparseDebugging ../

read -p "Now do it for real? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

rsync -avhzr lorell@slurm.ttic.edu:~/SparseDebugging ../


