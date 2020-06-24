#!/bin/bash

echo "Syncing up to Slurm. Dry Run..."

rsync -avhzrn ../SparseDebugging lorell@slurm.ttic.edu:~/

read -p "Now do it for real? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

rsync -avhzr ../SparseDebugging lorell@slurm.ttic.edu:~/


