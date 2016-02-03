# doc2case

Transcribe document-style I/O from the problem sets into the case format.

## Usage

```
$ ./doc2case.py -h
usage: doc2case.py [-h] [-m directory] [-t type] filename

positional arguments:
  filename      the IO to transcribe

optional arguments:
  -h, --help    show this help message and exit
  -m directory  path to the /dev/cases folder
  -t type       type of the IO
```

### Example

```
$ ./doc2case.py -m ~/Dropbox/WiC-BPC-Fa15/Solutions/dev/cases -t=corner ./Fa15ASkyFullofUnicorns.txt
```

A folder called "cases" will be created in the current working directory. A file will be created for each of your IO in ./Fa15ASkyFullofUnicorns.txt and the corresponding IO file in ~/Dropbox/WiC-BPC-Fa15/Solutions/dev/cases merged. The files are named from "problem1_corner.json" to "problem15_corner.json".

## IO Format

Reasonable format similar to those of the examples.

## Requirement

python 3.0 (probably, but 3.5 will do anyway)

## Note

Manual validating the output is required.
