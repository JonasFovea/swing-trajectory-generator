# build package

```shell
$ ./venv/bin/python setup.py sdist
$ ./venv/bin/python setup.py bdist_wheel sdist
```

# pip install locally
```shell
$ pip install -e .
```
with dev dependencies:
```shell
$ pip install -e .[dev]
```

# upload package
```shell
$ twine upload dist/*
```