#!/bin/bash

# FastVideo Monitoring Stack Startup Script

echo "🚀 Starting FastVideo Monitoring Stack..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}Port $1 is already in use${NC}"
        return 0
    else
        return 1
    fi
}

# Function to start Prometheus
start_prometheus() {
    echo "📊 Starting Prometheus..."
    if check_port 9090; then
        echo -e "${YELLOW}Prometheus already running on port 9090${NC}"
    else
        cd prometheus-3.5.0.linux-amd64
        nohup ./prometheus --config.file=prometheus.yml --storage.tsdb.path=./data --web.console.libraries=./console_libraries --web.console.templates=./consoles > ../prometheus.log 2>&1 &
        echo $! > ../prometheus.pid
        cd ..
        echo -e "${GREEN}✅ Prometheus started (PID: $(cat prometheus.pid))${NC}"
        echo "   📊 Web UI: http://localhost:9090"
    fi
}

# Function to start Grafana
start_grafana() {
    echo "📈 Starting Grafana..."
    if check_port 3000; then
        echo -e "${YELLOW}Grafana already running on port 3000${NC}"
    else
        sudo systemctl start grafana-server
        echo -e "${GREEN}✅ Grafana started${NC}"
        echo "   📈 Web UI: http://localhost:3000 (admin/admin)"
    fi
}

# Function to start Ray with metrics
start_ray() {
    echo "⚡ Starting Ray with metrics..."
    if check_port 8265; then
        echo -e "${YELLOW}Ray Dashboard already running on port 8265${NC}"
    else
        conda activate fv
        python start_ray_with_metrics.py
        echo -e "${GREEN}✅ Ray started with metrics${NC}"
        echo "   ⚡ Dashboard: http://localhost:8265"
        echo "   📊 Metrics: http://localhost:8080/metrics"
    fi
}

# Main execution
echo "🔧 Setting up monitoring stack..."

# Start services
start_prometheus
sleep 2
start_grafana
sleep 2
start_ray

echo ""
echo -e "${GREEN}🎉 Monitoring stack startup complete!${NC}"
echo ""
echo "📊 Access URLs:"
echo "   Prometheus:    http://localhost:9090"
echo "   Grafana:       http://localhost:3000 (admin/admin)"
echo "   Ray Dashboard: http://localhost:8265"
echo ""
echo "📝 Next steps:"
echo "   1. Configure Grafana data source: http://localhost:9090"
echo "   2. Import FastVideo dashboards"
echo "   3. Start your FastVideo Ray Serve app"
echo ""
echo "📄 Logs:"
echo "   Prometheus: ./prometheus.log"
echo "   Grafana:    journalctl -u grafana-server"
echo "" 