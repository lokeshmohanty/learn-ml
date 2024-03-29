{ pkgs ? import <nixpkgs> {}}:

pkgs.mkShell {
  packages = with pkgs.python311Packages; [ 
    # common
    pip jupyter numpy pandas matplotlib
    seaborn tqdm

    # utilities
    jupytext 

    # rl
    gymnasium pygame pybox2d

    # jax
    # equinox flax
    # tensorflow tensorflow-datasets

    # pytorch
    torch torchvision

    # others
    # dvc mlflow
    pydantic rich
  ];
  shellHook = ''
    # export PYGAME_DETECT_AVX2=1
    # source bin/activate
  '';
}
