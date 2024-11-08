import ray

ray.init(num_cpus=10)

@ray.remote
def square(x):
    return x * x

# Launch four parallel square tasks.
futures = [square.remote(i) for i in range(15)]

# Retrieve results.
print(ray.get(futures))