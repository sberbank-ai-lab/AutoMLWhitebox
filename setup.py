from setuptools import setup, find_packages

setup(
    name="autowoe",
    version="1.2",
    author="Anton Vakhrushev, Grigorii Penkin, Alexey Burlakov, Igor Myagkov",
    author_email='AGVakhrushev@sberbank.ru',
    license='MIT',
    description="Library for building interpreted model, based on logistic regression",
    keywords='autowoe',
    packages=find_packages(),
    include_package_data=True,
)
