
You need a GCC compiler and make tool to compile CLI version of the generator.

1. Go to `src/c_cli`.
2. Build it with `make` command.
3. Your program is in `dist/cli` directory, so you can run:
   ```bash
   ../../dist/cli/lorem-ipsum --help
   ```