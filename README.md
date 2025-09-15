# Lorem Ipsum

A deterministic Lorem Ipsum generator powered by a simple neural network.


## Features

- Deterministic output for given seed and parameters
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

You have multiple usage options:

* **On-line Web Generator**

  [`https://kildom.github.io/lorem-ipsum-nn/`](https://kildom.github.io/lorem-ipsum-nn/)

* **TypeScript/JavaScript**

  [`src/ts/README.md`](src/ts/README.md)

* **C/C++**

  Portable Library: [`src/c_lib/README.md`](src/c_lib/README.md)

  CLI Application: [`src/c_cli/README.md`](src/c_cli/README.md)

## Training

You can train custom models — for example, to support additional language stylizations.  
See the [`train/README.md`](train/README.md) file for detailed instructions.
