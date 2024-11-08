# Create a distributed computation system based on the RAY
https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#manual-setup-cluster

## Install the RAY
```pip install -U 'ray[default]'```

## Create a RAY Cluster

### Create a RAY Head
```ray start --head --node-ip-address '<your_ip_address>' --ray-debugger-external```

### Connect a RAY Node to the RAY Cluster
```ray start --address='<your_ip_address>'```

