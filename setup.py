from pathlib import Path

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def _load_requirements(path_dir: Path, comment_char: str = "#"):
    requirements_directory = path_dir / "requirements.txt"
    requirements = []
    with requirements_directory.open("r") as file:
        for line in file.readlines():
            line = line.lstrip()
            # Filter all comments
            if comment_char in line:
                line = line[: line.index(comment_char)]
            if line:  # If requirement is not empty
                requirements.append(line)
    return requirements


setup(
    name="deepfish",
    version="0.0.1",
    packages=find_packages(),
    install_requires=_load_requirements(THIS_DIR),
)
