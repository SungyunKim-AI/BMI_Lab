#!/bin/bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while read -r line; do
    # GPU 번호와 사용 중인 메모리 추출
    gpu_id=$(echo $line | awk '{print $1}' | tr -d ',')
    gpu_mem=$(echo $line | awk '{print $2}')

    echo "GPU ID: $gpu_id, Memory Used: ${gpu_mem}MiB"

    # 해당 GPU에서 실행 중인 프로세스 정보 가져오기
    nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=$gpu_id | while read -r pid; do
        # PID가 비어있지 않을 경우 도커 컨테이너 ID 및 이름 확인
        if [ ! -z "$pid" ]; then
            container_id=$(cat /proc/$pid/cgroup | grep 'docker' | sed 's/.*-//; s/^\(.\{12\}\).*/\1/')
            if [ -z "$container_id" ]; then
                echo "  PID: $pid, Not running inside a container"
            else
                container_name=$(docker ps --filter "id=$container_id" --format "{{.Names}}")
                echo "  PID: $pid, CID: $container_id, CName: $container_name"
            fi
        fi
    done
done
