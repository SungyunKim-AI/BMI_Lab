docker run -d \
    --name node_exporter \
    -p 9100:9100 \
    --network monitoring_backend \
    prom/node-exporter:v1.8.2