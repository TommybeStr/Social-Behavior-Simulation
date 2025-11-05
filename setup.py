from setuptools import setup, find_packages

setup(
    name="social_behavior_sim",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "verl.rollout_types": [
            "self_play = my_rollout:SelfPlayRollout",
        ],
    },
)
