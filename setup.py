from setuptools import setup, find_packages


setup(
    name='ddp_tokenization',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    scripts=['bin/ddp_tokenize_demo'],
    install_requires=['sphix', 'sphinx_rtd_theme', 'pytorch', 'numpy', 'fargv'
    ],
    entry_points={
        'console_scripts': [
            'script1 = my_package.module1:main',
            'script2 = my_package.module2:main',
        ]
    },
)