from setuptools import setup

package_name = 'final_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@email.com',
    description='Final project',
    license='TODO',
    entry_points={
        'console_scripts': [
            'detect3d = final_project.detect3d:main',
        ],
    },
)