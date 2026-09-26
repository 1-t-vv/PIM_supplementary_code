from fignet.simulator import LearnedSimulator


def __getattr__(name):
    if name == "Scene":
        from fignet.scene import Scene

        return Scene
    if name in ("rollout", "visualize_trajectory"):
        from fignet import utils

        return getattr(utils, name)
    raise AttributeError(name)
