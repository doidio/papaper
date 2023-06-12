# Papaper

academic papers embedding database

## For user

install [Java](https://www.java.com/download/)

install [Python 3.10](https://www.python.org/downloads/)

```shell
set-executionpolicy RemoteSigned

py -m venv venv
./venv/Scripts/activate.ps1

py -m pip install -U papaper
py -m papaper
```

## For dever

python 3.10 venv

```shell
briefcase dev
briefcase run
py -m twine upload dist/*
```
