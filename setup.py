from distutils.core import setup

setup(name='WarehouseEnv',
      version='1.0',
      description='Minimal warehouse environment for MAPF problems.',
      author='Evan Czyzycki',
      author_email='eczy3826@gmail.com',
      packages=['warehouse_env'],
      install_requires=[
          "gym==0.17",
      ]
     )