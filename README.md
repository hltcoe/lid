# Language Identification for Text

This repository is essentially a template for how *I* structure a new project.

            # ISO 639 (language)
            # ISO 15924 (script)
            # ISO 3166 (country)
            # ISO X (transliteration)


## Initializing and working with a new project

The following sequence:

```
git clone https://github.com/TomLippincott/new-project
mv new-project ${PROJECT\_NAME}
cd ${PROJECT\_NAME}
git remote remove origin
git remote add origin ${GIT\_URL}
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

The `${PROJECT\_NAME}` directory is now an (empty) experiment, with version-control, environment, and dependency management.  To work on and run the experiment, you simply run:

```
cd ${PROJECT\_NAME}
source venv/bin/activate
```

After days/weeks/months of work, you can just close the terminal, or just leave the environment with:

```
deactivate
```

## Common tasks

