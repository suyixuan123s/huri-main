""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231114osaka

"""
from pathlib import Path
import pyperclip

if __name__ == '__main__':
    def p(root_path):
        root_path = Path(root_path)
        folder_name = list(root_path.glob('session_*'))[-1]
        s = fr'docker run --rm -p 9090:9090 -v {folder_name}\metrics\prometheus\prometheus.yml:/etc/prometheus/prometheus.yml -v {root_path}\prom_metrics_service_discovery.json:/tmp/ray/prom_metrics_service_discovery.json prom/prometheus'
        print(s)
        pyperclip.copy(s)
        return "Text copied to clipboard."

    # http://localhost:3000/d/rayDefaultDashboard/default-dashboard
    p(r'G:\chen\ray_session')
