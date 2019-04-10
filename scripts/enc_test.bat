for %%i in (22, 27, 32, 37) do (
	h264enc.exe --qp %%i --output test_%%i.264 --input ..\sequence\foreman.cif 
)

pause