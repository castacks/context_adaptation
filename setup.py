from setuptools import find_packages, setup
import os
from glob import glob

package_name = "context_adaptation"
data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name), glob("launch/*.py")),
    (os.path.join("share", package_name, "config"), glob("config/*.yaml")),       
    (os.path.join("share", package_name, "assets", "context_clusters"), glob("assets/context_clusters/*.data")),
    (os.path.join("share", package_name, "assets", "context_configs"), glob("assets/context_configs/*.yaml")),
    (os.path.join("share", package_name, "assets", "cost_configs"), glob("assets/cost_configs/*.yaml")),
    (os.path.join("share", package_name, "assets", "costmap_configs"), glob("assets/costmap_configs/*.yaml")),
    (
        os.path.join("share", package_name, "assets", "dino_clusters", "not_turnpike_8"), 
        glob("assets/dino_clusters/not_turnpike_8/*.pt")
    ),
    (
        os.path.join("share", package_name, "assets", "dino_clusters", "VITB_not_turnpike_8"), 
        glob("assets/dino_clusters/VITB_not_turnpike_8/*.pt")
    ),    
    (os.path.join("share", package_name, "assets", "gp_params"), glob("assets/gp_params/*")),
]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="matthew",
    maintainer_email="matthew@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cost_publisher = context_adaptation.cost_publisher_updated:main",
            "dino_costmap_gp_speed_input_output = context_adaptation.dino_costmap_gp_speed_input_output:main",
            "cost_publisher_arl_sim = context_adaptation.cost_publisher_updated_arl_sim:main",
        ],
    },
)
