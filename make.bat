@ECHO OFF
REM ==============================================================================
REM Windows Batch Script for Sphinx Documentation Build
REM ==============================================================================
REM This script provides convenient commands for building Sphinx documentation
REM on Windows systems.
REM
REM Usage:
REM   make html      - Build HTML documentation
REM   make clean     - Remove all build artifacts
REM   make help      - Show all available targets
REM
REM The CI/CD pipeline uses sphinx-build directly, but this script is useful
REM for local development and testing on Windows.
REM ==============================================================================

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
REM Build directory - use _build for consistency with CI/CD pipeline
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
