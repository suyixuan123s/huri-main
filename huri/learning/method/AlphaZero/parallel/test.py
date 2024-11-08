""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230706osaka

"""

import ray


@ray.remote
def test():
    for i in range(100):
        breakpoint()
        print("TEST")


def main():
    ray.init(_node_ip_address='100.80.147.16', )

    ray.get(test.remote())

if __name__ == '__main__':
    main()
