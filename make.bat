@echo off
if "%1"=="up" (
    echo Creating k3d cluster...
    k3d cluster create --config cluster/k3d.yaml
    echo Deploying storage...
    kubectl apply -f k8s/storage/
    echo Deploying app...
    kubectl apply -f k8s/app/
) else if "%1"=="down" (
    echo Deleting k3d cluster...
    k3d cluster delete cloud-assignment
) else if "%1"=="train" (
    if "%SUBSET%"=="" (
        python training/train.py
    ) else (
        python training/train.py --subset %SUBSET%
    )
) else if "%1"=="test" (
    python -m unittest discover -s tests -p "test_*.py" -v
) else if "%1"=="lint" (
    python -m flake8 service/ worker/ training/ tests/ --max-line-length=120 --ignore=E501,W503
) else (
    echo Usage: make.bat [up^|down^|train^|test^|lint]
)

