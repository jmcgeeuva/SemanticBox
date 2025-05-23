#!/bin/bash
#SBATCH --job-name=preprocess-tech
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20 # number of cores/processors
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -A eng_viva
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=tkg5kq@virginia.edu
#SBATCH -a 12-21,29,33-37,42,45-47,50,53,59,84-85,87,99,101,111,116,123,129,134,142,147-177,179-223,225-275,277-338,340-576,578-837,839-883,885-912,914-1194,1197-1239,1241-1391,1393-1415,1417-1570,1573-1605,1607-1651,1654-1710,1712-1717,1719-1758,1760,1762-1764,1766-1921,1923-2024,2026-2107,2109-2163%100

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

if [[ -n "${SLURM_ARRAY_JOB_ID}" ]]; then
    mkdir -p "$LOG_DIR/$SLURM_ARRAY_JOB_ID"
else
    now=$(date +"%y%m%d-%H%M%S")
    mkdir -p "$LOG_DIR/$now"
fi

# configure log file path
# Check if SLURM_ARRAY_JOB_ID is set and not empty
if [[ -n "${SLURM_ARRAY_JOB_ID}" ]]; then
    now=$(date +"%y%m%d")
    logpath="${LOG_DIR}/$SLURM_ARRAY_JOB_ID/logs-$now-${SLURM_ARRAY_JOB_ID}"
    mkdir -p $logpath
    logfile="$logpath/${SLURM_ARRAY_TASK_ID}.out"
else
    # Use the last argument as the log file if SLURM_ARRAY_JOB_ID is not set
    now=$(date +"%y%m%d-%H%M%S")
    logfile="${LOG_DIR}/$now/logs-$now.out"
fi

source /home/tkg5kq/.bashrc > "${logfile}" 2>&1
source activate bound >> "${logfile}" 2>&1

echo "Running $@ with ID ${SLURM_ARRAY_TASK_ID} ..."

python preprocess_new_annotations.py technology --vid_num ${SLURM_ARRAY_TASK_ID} >> "${logfile}" 2>&1

sleep 45