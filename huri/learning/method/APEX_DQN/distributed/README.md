# 学习运行
1. 在[params](params)增加运行参数
2. 修改`config_path` in [main.py](main.py)
3. `ray start --head` 

# 测试运行
- **测试Env**: [test.py](..%2F..%2F..%2Fenv%2Frack_v3%2Ftest%2Ftest.py)
- **测试运行**: [eval_playground.py](test%2Feval_playground.py)

# Ray 信息
- Dead process in Ray: `ray kill --all` 
- Dead process does not be deleted: `ray stop`


# 
docker run -p 9090:9090 -v C:\Users\WRS\AppData\Local\Temp\ray\session_2023-11-14_22-11-31_386055_3112\metrics\prometheus\prometheus.yml:/etc/prometheus/prometheus.yml -v C:\Users\WRS\AppData\Local\Temp\ray\prom_metrics_service_discovery.json:/tmp/ray/prom_metrics_service_discovery.json prom/prometheus