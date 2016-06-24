from setuptools import setup

setup(name='gp_emu',
      version='0.1',
      description='Gaussian Process Emulator',
      url='http://github.com/samcoveney/GP_emu',
      author='Sam Coveney',
      author_email='coveney.sam@gmail.com',
      license='GPL-3.0+',
      packages=['gp_emu', 'gp_emu/design_inputs'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'future',
      ],
      zip_safe=False)
