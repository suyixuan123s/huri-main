""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from huri.learning.method.APEX_DQN.distributed.pipeline import EpsilonScheduler

    matplotlib.use("TkAgg")

    decay_rate = 0.0001
    total_steps = 1 / decay_rate
    init_epsilon = 1
    final_epsilon = .1

    es = EpsilonScheduler(init_epsilon, final_epsilon, decay_rate)

    results = []
    for i in range(int(total_steps) + 1000):
        results.append([i, es.get_epsilon()])
        es.step()
    es.reset()
    results.append([total_steps + 1000, es.get_epsilon()])

    steps = [row[0] for row in results]
    epsilon_values = [row[1] for row in results]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, epsilon_values, label='Epsilon Decay')

    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay over Steps')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
