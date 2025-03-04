{
  description = "flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs, ... }: let

    pkgs = nixpkgs.legacyPackages."x86_64-linux";

  in {
    devShells.x86_64-linux.default = pkgs.mkShell {

      packages = with pkgs; [
        (python3.withPackages (p: with p; [ jupyter-book matplotlib numpy pip ]))
      ];
    };
  };
}
