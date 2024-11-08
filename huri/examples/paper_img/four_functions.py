import numpy as np

from huri.components.utils.matlibplot_utils import Plot
from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot

color = {
    1: "purple",
    2: "gray",
    3: "blue",
    4: "pink",
    "alpha": .7,
    "line_color": "k"
}

rack_size = (5, 10)
num_classes = 3
observation_space_dim = (num_classes * 2, *rack_size)
action_space_dim = np.prod(rack_size) ** 2
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=False,
                         is_goalpattern_fixed=False)
pattern = env.reset()

# init a plot
plot = Plot(w=3.5, h=5, is_axis=False, dpi=300)
drawer = RackStatePlot(pattern, color, plot=plot)

fig = drawer.plot_bg()
plot.save_fig('N')

# Nc
Nc = np.array([[1, 1, 1, 3, 3, 3, 3, 2, 2, 2, ],
               [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, ],
               [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, ],
               [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, ],
               [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, ]])
#
plot = Plot(w=3.5, h=5, is_axis=False, dpi=300)
drawer = RackStatePlot(Nc, color, plot)
#
drawer.plot_bg()
plot.save_fig('G')

for i in range(3):
    env.reset_state(pattern)
    cc = env.step(env.sample())[0]
    plot = Plot(w=3.5, h=5, is_axis=False, dpi=300)
    fig = RackStatePlot(cc, color, plot).plot_bg()
    plot.save_fig(f'H_{i}')

Nc = np.array([[1, 1, 3, 3, 2, 2, ],
               [1, 1, 3, 3, 2, 2, ],
               [1, 1, 3, 3, 2, 2, ],
               [1, 1, 3, 3, 2, 2, ]])

for i in range(3):
    cc = Nc.copy()
    for _ in range(np.random.randint(1, 10)):
        cc[tuple(np.random.randint(Nc.shape))] = 0
    plot = Plot(w=3.5, h=5, is_axis=False, dpi=300)
    fig = RackStatePlot(cc, color, plot).plot_bg()
    plot.save_fig(f'Rg_{i}')
