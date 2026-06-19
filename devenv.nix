{ pkgs, lib, config, inputs, ... }:

{
  # 1. Define native system binaries (non-python packages)
  packages = with pkgs; [
    ruff
    # 'ty' isn't a standard top-level nixpkgs binary; 
    # if it's a typo for 'typer-cli' or something similar, add it here.
  ];

  # 2. Enable Python and configure its packages
languages.python = {
    enable = true;
    # devenv defaults to a stable python version. If you specifically need 3.13:
    # version = "3.13";

    package = pkgs.python3.withPackages ( ps: with ps; [
      numpy
      pandas
      torch
      scipy
      scikit-learn 
      sentencepiece
      sentence-transformers
      pyttsx3
      keras
      tensorflow

      # espeak
      sacrebleu
      rouge-score
      seqeval
      sklearn-compat
      speechrecognition
      nltk
      spacy
      transformers
      spacy-transformers
      hf-xet
      rich
      notebook
      tensorflow
      dm-tree
      matplotlib
      seaborn
      selenium
      
      # requests
      # pymc
      # arviz
      # beautifulsoup4
      # pymongo
      # plotly
      # toolz
    ]);

    # Enables an automated virtual environment (.devenv/state/venv) 
    # and allows you to run pip install for things not packaged in Nix
    venv = {
      enable = true;
      quiet = true;
      requirements = ''
        sklearn-crfsuite
        SpeechRecognition
      '';
    };
  };
  env.LD_LIBRARY_PATH = "${pkgs.espeak-ng}/lib";

  # 3. Handle post-shell initializations
  enterShell = ''
  '';
}
