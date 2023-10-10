#coding=utf-8
#cxsetup.py代码
from cx_Freeze import setup, Executable
setup(
    name="test",
    version="1.0",
    description="Test application",
    author="gilfoyle",
    executables=[Executable("handWritingNumberRecognizationApplication.py")]
)
