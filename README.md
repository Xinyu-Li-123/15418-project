# Example CMake Project for CPP + CUDA

## Prerequisites

You would want to first add `~/bin` to path. This allows us to add / overwrite binary dependencies manually.

- ninja: Ninja is a light-weight, efficient build system. CMake uses it. You can just download binary and place it under user's bin directory at `~/bin/ninja`

- gcc-11: We need to use gcc-11, since GHC uses cuda 11.7, and it can't work with higher version of gcc. GHC uses gcc-13 by default. To overwrite it, you can create symbolic link `~/bin/gcc` that points to to `/usr/bin/gcc-11`

## Build dev and run tests

```bash
# Clear previous builds
rm -rf build/dev

cmake --preset dev
cmake --build --preset dev
# The binary output is at 
./build/dev/peakfinder_cli

# You can also run unit test with
ctest --preset dev
```

## Build for release

```bash
# Clear previous builds
rm -rf build/release

cmake --preset release
cmake --build --preset release
# The binary output is at 
./build/release/peakfinder_cli
```

## LSP Config (Clangd)

You can link `./build/dev/compile_commands.json` to `./compile_commands.json` to get clangd working. We also have a `.clangd` config file that ignores some cuda stuff since clangd can't fully work with cuda code.
