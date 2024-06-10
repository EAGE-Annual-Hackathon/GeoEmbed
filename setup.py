from distutils.core import setup


setup(
    name='geoembed',
    packages=['geoembed'],
    url='',
    description='',
    entry_points={
        "console_scripts": ["{0}={0}.__main__:main".format('geoembed')]}
)
