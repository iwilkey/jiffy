# JIFF-Y, the AI GIF generator.
#### Developer, Ian Wilkey (iwilkey)
---
JIFF-Y is an open-source GIF generator utilizing a robust Generative Adversarial Network.

Similar to OpenAI's DALL-E, which served as the primary inspiration for this project, JIFF-Y aims to produce authentic GIFs suitable for any given context. Note that, because Natural Language Processing (NLP) is a relatively recent domain in Machine Learning research, the preliminary phase of the JIFF-Y project strives to generate a randomized GIF based on its observations from the TGIF dataset.

JIFF-Y's future iterations will entail the inclusion of a web-based Graphical User Interface (GUI) for user interaction, and the capacity to generate GIFs based on natural language prompts. Given that this project is open-source, collaboration is critical for this phase, as well as discussions pertaining to the user interface and its user-friendliness.

---

## Getting Started

As this project exclusively utilizes Python, the initial step would be to install Python 3.11. You may download the appropriate version for your operating system by visiting the official Python website: https://www.python.org/downloads/.

Now you're ready to test and collaborate with JIFF-Y. It is advisable to create an empty directory and employ a Python environment for optimal usage.

``` bash
mkdir jiffy-test
cd jiffy-test
git clone https://github.com/iwilkey/jiffy
```

Next, the creation of a Python environment is recommended to encapsulate all dependencies in one place.

``` bash
# In jiffy-test/
python3.11 -m venv env
source env/bin/activate
```

In your Python environment, install all of the dependencies using the "requirements.txt" file.

``` bash
cd jiffy
# Upgrade pip if needed. This is recommended, just in case.
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Collecting GIFs from TGIF Dataset

Given that the project heavily depends on the TGIF dataset, a copy of the data and a Python script have been included for automatic download of a designated number of GIFs from the training set. You may utilize this script to effortlessly populate your directories with numerous like GIFs for experimentation.

Proceed to the directory that stores TGIF and execute the "Get" software. Engage with the runtime to indicate the target phrase and preferred quantity of GIFs and the designated saving directory (default = jiffy/data/target).

``` bash
# From the project root (jiffy/).
cd data
python3.11 dataq.py
```

Upon successful completion of the program's download and preservation of the TGIF GIFs, you can employ them to experiment with JIFF-Y.

---
