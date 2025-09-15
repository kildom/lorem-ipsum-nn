
# Lorem Ipsum Models Training

The `train` directory contains scripts for training new models.

## Dependencies

Before you start, I suggest creating and activating a Python virtual environment.

Install python dependecies from the `requirements.txt` file:

```shell
pip install -r requirements.txt
```

## Running the script

For command line options run:

```shell
python main.py --help
```

## Creating new language stylization

To create a new language stylization, create a new `lang_**.py` file,
where `**` is an ISO language code. Use `lang_en.py` as an example.
It also contains comments that will help you with your file.

When you have your file ready, you can start training:

```shell
python main.py **
```

(where `**` is your language code)

It will train your model and place output files in the `models` directory.
You can now build the project with your model in it.