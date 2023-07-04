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
                  hasktorch-skeleton = ./hasktorch-skeleton; });};})
      ];

      pkgs = import inputs.nixpkgs { inherit system; overlays = [ overlay ]; };

      hlib = pkgs.haskell.lib;

      haskPkgs = __mapAttrs (_: ps: ps.haskellPackages) pkgs.hasktorchPkgs;

    in

      {

        inherit pkgs overlay;

        packages.${system} = {
          default = haskPkgs.${hasktorch-device}.hasktorch-skeleton;
          hasktorch = haskPkgs.${hasktorch-device}.hasktorch;
          hasktorch-skeleton = __mapAttrs (_: hpkgs: hpkgs.hasktorch-skeleton) haskPkgs;
        };

        devShell.${system} =
          let
            hpkgs = haskPkgs.${hasktorch-device};
          in
            hpkgs.shellFor {

              packages = p: with p; [
                hasktorch-skeleton
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
