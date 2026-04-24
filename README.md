# 15-418 Project: C++ Implementation of GpJSON

Project website: <https://xinyu-li-123.github.io/15418-project-site/>

## How to run

We provide a driver program to run our implementation of GpJSON under `apps/`. It is a command line program that takes in a toml config file and command line flags to configure execution of GpJSON.

You can compile the program with a preset, and run the driver program with

```bash
cmake --preset <preset>
cmake --build --preset <preset>
./build/<preset>/apps/gpjson_driver --config <path to config file>
```

We have provided default config files for different queries used in GpJSON paper under `config/`.

## TODO

- [ ] Consider manually pad the file size to multiple of 8 / 16 / etc, so that we don't need to branch to check index against file size
