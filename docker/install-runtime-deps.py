"""Emit Docker runtime requirements from pyproject.toml.

Docker installs torch from the PyTorch CPU wheel index first. Keeping torch out of
this generated requirements file prevents pip from replacing it with the default
CUDA-enabled PyPI build, which is both huge and unnecessary for this service.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


PYPROJECT = Path("pyproject.toml")
OUT = Path("/tmp/docvault-runtime-requirements.txt")


def package_name(requirement: str) -> str:
    requirement = requirement.split(";", 1)[0].strip().lower()
    requirement = requirement.split("[", 1)[0]
    for operator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        requirement = requirement.split(operator, 1)[0]
    return requirement.strip()


def main() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    dependencies = data["project"]["dependencies"]
    runtime_dependencies = [dep for dep in dependencies if package_name(dep) != "torch"]
    OUT.write_text("\n".join(runtime_dependencies) + "\n")


if __name__ == "__main__":
    main()
