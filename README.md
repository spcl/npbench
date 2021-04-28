# NPBench

## Quickstart

To install NPBench, simply execute:
```
./setup.py
```
You can then run benchmarks with:
```
python run_benchmark.py -b <benchmark> -f <framework>
```
The available benchmarks are listed in the `bench_info` folder. The available frameworks are listed in the `framework_info` folder. For example, to run ADI with NumPy, execute the following:
```
python run_benchmark.py -b adi -f numpy
```
Please note that the NPBench setup only installs NumPy. To run benchmarks with other frameworks, you have to install them separately.

## Adding Benchmarks

TODO

## Adding Frameworks

TODO
