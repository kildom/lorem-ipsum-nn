# C/C++ Library

The library is provided in source form.
You can include the source and header files into your project
or compile the library on your own.

## Source files

List of source files that you need:

* `src/c_lib/src/*.c` - generator main source file and stylization models definitions.
  If you disabled some stylization with the configuration options, you don't need to compile and link related file.

Include directories:

* `src/c_lib/include` - contains public API header file `lorem-ipsum.h` and internal header files.

## Configuration

You can select which language stylization models you want to compile in. By default, all models are compiled. You can choose ONE way of configuration:

* define one or more `LOREM_IPSUM_xx_DISABLED` will disable specific laguage stylizations, all others will be enabled.

* define one or more `LOREM_IPSUM_xx_ENABLED` will enable specific laguage stylizations, all others will be disabled.

For example:

 * if you want just latin stylization, define `LOREM_IPSUM_LA_ENABLED`.

 * if you don't need stylization that use alphabet not based on latin, define `LOREM_IPSUM_EL_ENABLED` and `LOREM_IPSUM_UK_ENABLED`.