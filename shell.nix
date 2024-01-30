# save this as shell.nix
# { pkgs ? import <nixpkgs> {}}:
{ pkgs ? import <nixpkgs> {}}:

pkgs.mkShell {
  packages = with pkgs.python311Packages; [ 
    pip jupyter # setuptools
    dvc mlflow
    equinox # flax
    gymnasium pygame
    tensorflow # tensorflow-datasets
    pydantic rich
    torch torchvision
  ];
  shellHook = ''
    export PYGAME_DETECT_AVX2=1
    source bin/activate
  '';
}
