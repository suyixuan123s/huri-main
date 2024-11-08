import numpy as np

from huri.components.utils.matlibplot_utils import Plot
from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot

color = {
    1: "red",
    2: "green",
    3: "blue",
    4: "pink",
    "alpha": .7,
    "line_color": "k"
}

rack_size = (11, 10)
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

rack_state = np.zeros(rack_size)
# init a plot
plot = Plot(w=5, h=5, is_axis=False, dpi=300)
drawer = RackStatePlot(rack_state, color, plot=plot)

fig = drawer.plot_bg()
plot.save_fig('two_rack')

