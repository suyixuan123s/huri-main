from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
import huri.core.file_sys as fs

DQNSolver(
    model_path=fs.workdir_learning / "run" / f"dqn_2022_01_08_21_13_41" / "model" / "model_5076000-5078000.pth",
    num_tube_classes=3,
    rack_size=(3, 5),
)
