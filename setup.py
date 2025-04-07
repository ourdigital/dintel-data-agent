"""
데이터 분석 에이전트 패키지 설정.
"""

from setuptools import setup, find_packages

# 의존성 패키지 목록 읽기
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="data-analysis-agent",
    version="0.1.0",
    author="Our Digital Team",
    author_email="team@ourdigital.example.com",
    description="다양한 소스에서 데이터를 수집, 처리, 분석하고 시각화하는 맞춤형 데이터 분석 에이전트",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ourdigital/data-analysis-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "data-analysis-agent=main:main",
        ],
    },
)
