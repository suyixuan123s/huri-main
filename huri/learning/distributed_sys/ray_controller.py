""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230706osaka

"""
import dataclasses
from typing import List, Optional, Mapping, Any

@dataclasses.dataclass
class RayRuntimeEnv:
    """Representation of a runtime environment."""

    pip: str
    working_dir: str


@dataclasses.dataclass
class RayJob:
    """Representation of a Tpu-based Ray Job."""

    entrypoint: str
    working_dir: str
    pip_installs: List[str] = dataclasses.field(default_factory=list)
    env_vars: Mapping[str, str] = None
    entrypoint_resources: Mapping[str, int] = None

    def to_ray_job(self) -> Mapping[str, Any]:
        return dict(
            entrypoint=self.entrypoint,
            runtime_env=dict(
                working_dir=self.working_dir,
                pip=self.pip_installs,
                env_vars=self.env_vars,
            ),
            entrypoint_resources=self.entrypoint_resources,
        )


if __name__ == '__main__':
    pass
