{
  description = "hasktorch-skeleton in Nix";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";

  inputs.hasktorch.url = github:hasktorch/hasktorch;
  inputs.hasktorch.flake = false;

  inputs.tokenizers.url = github:hasktorch/tokenizers/9d25f0ba303193e97f8b22b9c93cbc84725886c3;
  inputs.tokenizers.flake = false;

  inputs.typelevel-rewrite-rules.url = github:hasktorch/typelevel-rewrite-rules/4176e10d4de2d1310506c0fcf6a74956d81d59b6;
  inputs.typelevel-rewrite-rules.flake = false;

  inputs.type-errors-pretty.url = github:hasktorch/type-errors-pretty/32d7abec6a21c42a5f960d7f4133d604e8be79ec;
  inputs.type-errors-pretty.flake = false;

  inputs.inline-c.url = github:fpco/inline-c/2d0fe9b2f0aa0e1aefc7bfed95a501e59486afb0;
  inputs.inline-c.flake = false;

  outputs = inputs:

    let
      system = "x86_64-linux";

      overlay = self: super: {
        libtorch = self.callPackage "${inputs.hasktorch}/nix/libtorch.nix" {
          cudaSupport = false;
          device = "cpu";
        };
        haskell = let
            packageOverrides = hself: hsuper: {
              tokenizers = hself.callCabal2nix "tokenizers" "${inputs.tokenizers}/bindings/haskell/tokenizers-haskell" {};
              typelevel-rewrite-rules = hself.callCabal2nix "typelevel-rewrite-rules" inputs.typelevel-rewrite-rules {};
              type-errors-pretty = hself.callCabal2nix "type-errors-pretty" inputs.type-errors-pretty {};
              inline-c = hself.callCabal2nix "inline-c" "${inputs.inline-c}/inline-c" {};
              inline-c-cpp = hself.callCabal2nix "inline-c-cpp" "${inputs.inline-c}/inline-c-cpp" {};

              codegen = hself.callCabal2nix "codegen" "${inputs.hasktorch}/codegen" {};
              libtorch-ffi = self.haskell.lib.compose.appendConfigureFlag "--extra-include-dirs=${self.libtorch.dev}/include/torch/csrc/api/include"
                (hself.callCabal2nix "libtorch-ffi" "${inputs.hasktorch}/libtorch-ffi" {
                  torch = self.libtorch;
                  c10 = self.libtorch;
                  torch_cpu = self.libtorch;
                });
              libtorch-ffi-helper = hself.callCabal2nix "libtorch-ffi-helper" "${inputs.hasktorch}/libtorch-ffi-helper" {};
              hasktorch = hself.callCabal2nix "hasktorch" "${inputs.hasktorch}/hasktorch" {};
              examples = hself.callCabal2nix "examples" "${inputs.hasktorch}/examples" {};
              experimental = hself.callCabal2nix "experimental" "${inputs.hasktorch}/experimental" {};

              hasktorch-skeleton = hself.callCabal2nix "hastorch-skeleton" ./hasktorch-skeleton {};
            };
          in super.haskell // { inherit packageOverrides; };
        haskellPackages = self.haskell.packages.ghc924;
      };

      pkgs = import inputs.nixpkgs { inherit system; overlays = [ overlay ]; };


    in

      {

        inherit pkgs overlay;

        packages.${system} = with pkgs; {
          default = haskellPackages.hasktorch-skeleton;
          hasktorch = haskellPackages.hasktorch;
          hasktorch-skeleton = haskellPackages.hasktorch-skeleton;
        };

        devShell.${system} = with pkgs; haskellPackages.shellFor {
          packages = p: with p; [
            hasktorch-skeleton
          ];
          buildInputs =
            (with haskellPackages;
            [ haskell-language-server
              threadscope
            ]) ++
            [
              ghcid.bin
              cabal-install
            ];
        };
      };

}
