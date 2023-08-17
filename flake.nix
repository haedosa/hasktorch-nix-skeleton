{
  description = "hasktorch-skeleton in Nix";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";

  inputs.hasktorch-nix.url = "github:asgard-labs/hasktorch-nix";

  outputs = inputs:

    let

      inherit (inputs.nixpkgs) lib;

      system = "x86_64-linux";

      ghc-name = "ghc924";

      hasktorch-device = "cuda-11"; # There are three variants "cpu" "cuda-10" "cuda-11"

      overlay = lib.composeManyExtensions [
        inputs.hasktorch-nix.overlay
        (self: super: { haskellPackages = self.haskell.packages.${ghc-name}; })
        (self: super: {
          haskell = super.haskell // {
            packageOverrides =
              lib.composeExtensions
                super.haskell.packageOverrides
                (self.haskell.lib.packageSourceOverrides {
                  hasktorch-skeleton = ./hasktorch-skeleton;
                  hasktorch-examples = ./hasktorch-examples;
                });};})
      ];

      pkgs = import inputs.nixpkgs { inherit system; overlays = [ overlay ]; };

      hlib = pkgs.haskell.lib;

      haskPkgs = __mapAttrs (_: ps: ps.haskellPackages) pkgs.hasktorchPkgs;

      mk-hasktorch-packages = hasktorch-device: rec {
        hasktorch-and-examples = pkgs.symlinkJoin {
          name = "hasktorch-and-examples";
          paths = [
            hasktorch
            hasktorch-examples
            hasktorch-skeleton
          ];
        };
        hasktorch = haskPkgs.${hasktorch-device}.hasktorch;
        hasktorch-examples = haskPkgs.${hasktorch-device}.hasktorch-examples;
        hasktorch-skeleton = haskPkgs.${hasktorch-device}.hasktorch-skeleton;
      };

    in

      {

        inherit pkgs overlay;

        packages.${system} = rec {
          default = (mk-hasktorch-packages "cuda-11").hasktorch-and-examples;
        }
        // (mk-hasktorch-packages "cuda-11")
        // (__mapAttrs (device: _: mk-hasktorch-packages device) haskPkgs);

        devShell.${system} =
          let
            hpkgs = haskPkgs.${hasktorch-device};
          in
            hpkgs.shellFor {

              packages = p: with p; [
                hasktorch-skeleton
                hasktorch-examples
              ];

              buildInputs =
                (with hpkgs;
                  [ haskell-language-server
                    threadscope
                  ]) ++
                (with pkgs;
                  [
                    ghcid.bin
                    cabal-install
                  ]);
            };

      };

}
