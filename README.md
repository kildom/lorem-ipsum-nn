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

* **On-line Web Generator** - go to [`https://kildom.githib.io/lorem-ipsum-nn/`](https://kildom.github.io/lorem-ipsum-nn/).
* **Portable C/C++ Library** - see [`src/c_lib/README.md`](src/c_lib/README.md).
* **JavaScript Package** - see [`src/js/README.md`](`src/python/README.md`), also available on [`npm`](https://www.npmjs.com/package/lorem-ipsum-nn), or as a standalone script from [`releases`](https://github.com/kildom/lorem-ipsum-nn/releases), tested with browsers, Node, and Deno. Includes TypeScript declarations.
* **CLI Program** - see [`src/c_cli/README.md`](src/c_cli/README.md), you can also download precompiled binaries from [`releases`](https://github.com/kildom/lorem-ipsum-nn/releases).

TODO:

* **Python Package** - see [`src/python/README.md`](`src/python/README.md`), or install with [`pip`](https://pypi.org/project/lorem-ipsum-nn/)

## Training

You can train your own models, for example, to add new language stylization.
See [`train/README.md`](train/README.md) for details.
