# Lorem Ipsum

A deterministic Lorem Ipsum generator powered by a simple neural network.


## Features

- Deterministic output for given seed and parameters
- Generates consistent text
- Can stylize generated text to resemble different languages (e.g. English, Polish)
- Built with a minimal neural network architecture
- Uses only integer operations, making it fast and portable


## Example output

> ### la - Latin stylization
>
> Lorem ipsum dolor sit amet, tencum estulum per ansi tri hectatus tulti
> netiunt. Sene irdo. Code catur qui cort auxus, et costi, quae in verit
> ve segoriti. Et core et petur et tult culus at cotam der derit ante
> paris tace. Quae mancit. Diecit abi an antur fersis, rodia vacitis et
> quim flodati it desore senis in cant, inse corse folle non, ad. Acte
> insa no balit, agrerit setiis ex lollum in.

> ### pl - Polish stylization
>
> Sto latie dnie się z zapolko, się jem czac mowyle, stał gareiłam.
> Porzyła z pam, zwiedziała się. Wnie do w moje nie. Chwiała pochodzią
> mrobienie stowi że, co się sięcze. Pocie tem szy mierygrzedzie w
> zaczę w pody ponałam. Postnie nie dostem osto omnie wie mie nie
> porwiedała zadzi nie się mo podzie.

## Usage

You have a lot of usage options, since the generator was ported on multiple platforms.
On each platform it gives the same deterministic output.

* **Portable C/C++ Library** - see [`impl/c_lib/README.md`](impl/c_lib/README.md).
* **CLI Program** - see [`impl/c_cli/README.md`](impl/c_cli/README.md), you can also download precompiled binaries from [`releases`](https://github.com/kildom/lorem-ipsum-nn/releases).

TODO:

* **On-line Web Generator** - go to [`https://kildom.githib.io/lorem-ipsum-nn/`](https://kildom.github.io/lorem-ipsum-nn/).
* **Python Package** - see [`impl/python/README.md`](`impl/python/README.md`), or install with [`pip`](https://pypi.org/project/lorem-ipsum-nn/)
* **JavaScript Package** - see [`impl/js/README.md`](`impl/python/README.md`), also available on [`npm`](https://www.npmjs.com/package/lorem-ipsum-nn), or as a standalone script from [`releases`](https://github.com/kildom/lorem-ipsum-nn/releases), tested with browsers, Node, and Deno. Includes TypeScript declarations.

## Training

You can train your own models, for example, to add new language stylization.
See [`train/README.md`](train/README.md) for details.
