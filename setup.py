from setuptools import setup, find_packages

setup(
    name="fastvideo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        # For decord, you might need to specify a different version or source
        # "decord>=0.6.0",  # Uncomment and adjust if needed
    ],
    entry_points={
        'console_scripts': [
            'fastvideo=fastvideo.v1.entrypoints.cli.main:main',
        ],
    },
) 