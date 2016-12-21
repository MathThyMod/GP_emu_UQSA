from setuptools import setup

setup(name='gp_emu_uqsa',
      version='1.0',
      description='Gaussian Process emulation for Uncertainty Quantification and Sensitivity Analysis',
      url='http://github.com/samcoveney/GP_emu_UQSA',
      author='Sam Coveney',
      author_email='coveney.sam@gmail.com',
      license='GPL-3.0+',
      packages=['gp_emu_uqsa', 'gp_emu_uqsa/design_inputs', 'gp_emu_uqsa/sensitivity'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'future',
      ],
      zip_safe=False)
