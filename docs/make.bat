@echo off

SET SOURCEDIR=source
SET BUILDDIR=_build
@REM Should be equivalent to the ?= in the Makefile
IF "%SPHINXBUILD%"=="" (SET SPHINXBUILD=sphinx-build)
@REM We currently don't have any other SPHINXOPTS
@REM IF "%SPHINXOPTS%"=="" (SET SPHINXOPTS=)

IF /I "%1"=="help" GOTO help
IF /I "%1"=="apidocs" GOTO apidocs

@REM Instead of the % from the Makefile, this should do the same
GOTO default

:help
	@%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	GOTO :EOF

:default
	@%SPHINXBUILD% -M "%1" "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	GOTO :EOF

:apidocs
	sphinx-apidoc --templatedir=source/templates -e -d 6 -f -o source/api ../mala
	GOTO :EOF
