#!/bin/bash
TEXTFILE_COLLECTOR_DIR="/home/sungyunkim_970623/monitoring/metric/gpu_mem_stats"
TEMP_FILE="$TEXTFILE_COLLECTOR_DIR/gpu_metrics.prom.$$"

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while read -r line; do
    gpu_id=$(echo $line | awk '{print $1}' | tr -d ',')
    gpu_mem=$(echo $line | awk '{print $2}' | tr -d ',')
    gpu_util=$(echo $line | awk '{print $3}')
    
    # Default output for GPUs
    echo "gpu_memory_used{gpu_id=\"$gpu_id\"} $gpu_mem" >> "$TEMP_FILE"
    echo "gpu_utilization{gpu_id=\"$gpu_id\"} $gpu_util" >> "$TEMP_FILE"

    # Get PIDs for the current GPU
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=$gpu_id)
    
    if [ -z "$pids" ]; then
        # Handle GPUs with no processes
        echo "gpu_pid{gpu_id=\"$gpu_id\", gpu_cid=\"none\", gpu_cname=\"none\"} $pid" >> "$TEMP_FILE"
    else
        # Process each PID
        echo "$pids" | while read -r pid; do
            if [ ! -z "$pid" ]; then
                container_id=$(cat /proc/$pid/cgroup 2>/dev/null | grep 'docker' | sed 's/.*-//; s/^\(.\{12\}\).*/\1/')
                container_name="none"
                if [ ! -z "$container_id" ]; then
                    container_name=$(docker ps --filter "id=$container_id" --format "{{.Names}}")
                fi

                echo "gpu_pid{gpu_id=\"$gpu_id\", gpu_cid=\"$container_id\", gpu_cname=\"$container_name\"} $pid" >> "$TEMP_FILE"
            fi
        done
    fi
done

mv "$TEMP_FILE" "$TEXTFILE_COLLECTOR_DIR/gpu_metrics.prom"