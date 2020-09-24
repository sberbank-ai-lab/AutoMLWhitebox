from setuptools import setup, find_packages

# with open('./requirements.txt') as f:
#     required = f.read().splitlines()

setup(
    name="autowoe",
    version="1.1",
    author="Anton Vakhrushev, Grigorii Penkin, Alexey Burlakov",
    author_email='AGVakhrushev@sberbank.ru',
    license='MIT',
    description="Library for building interpreted model, based on logistic regression",
    keywords='autowoe',
#     install_requires=required,
    packages=find_packages(),
    include_package_data=True,
)

