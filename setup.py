#!/usr/bin/env python3
"""
setup.py - Package installation configuration for lmagi
(c) Gregory L. Magnusson MIT license 2024
easyAGI - easy augmented generative intelligence
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='lmagi',
    version='1.0.0',
    description='easyAGI - Multi-model LLM with automind reasoning',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Gregory L. Magnusson',
    author_email='',
    url='https://github.com/yourusername/lmagi',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='agi llm ai reasoning automation openai groq ollama',
    project_urls={
        'Documentation': 'https://rage.pythai.net',
        'Source': 'https://github.com/yourusername/lmagi',
    },
    entry_points={
        'console_scripts': [
            'lmagi=lmagi:main',
        ],
    },
    include_package_data=True,
    package_data={
        'lmagi': ['gfx/*'],
    },
    zip_safe=False,
)
