@ECHO OFF

PUSHD %~dp0

REM base task model
git clone https://github.com/Kahsolt/PyTorch_CIFAR10
REM adv attack
git clone https://github.com/Harry24k/adversarial-attacks-pytorch

POPD

ECHO Done!
ECHO.

PAUSE
