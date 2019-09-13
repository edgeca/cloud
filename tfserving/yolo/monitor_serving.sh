#!/bin/bash
tensorflow_model_server --port=8500 --model_config_file=/models/models.config & export app_pid=$!

# Monitoring Loop
run_start_time=`date +%s`
log_file_name="/tfs-logs/run_log_${run_start_time}.csv"
while ps -p ${app_pid}>/dev/null; do
  cpu_mem=`top -n1 -b -p $app_pid | tail -1`
  cpu=`echo $cpu_mem | awk '{print $9}'`
  mem=`echo $cpu_mem | awk '{print $10}'`
  timestamp=`date +%s`
  echo "${timestamp},${cpu},${mem}" >> "${log_file_name}"
  sleep 10
done