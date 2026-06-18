{
  # Experimental devShell to benchmark Intel-GPU (Iris Xe, Level Zero) embedding
  # via torch-XPU + sentence-transformers, vs CPU. Supplies the Level Zero
  # loader + Intel L0 GPU driver + a C/C++ runtime for the manylinux torch-xpu
  # wheels (nix-ld resolves the rest). torch-xpu itself is installed with uv
  # from the PyTorch XPU wheel index inside the shell.
  description = "Intel Iris Xe XPU embedding benchmark devShell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      # The Intel Level Zero GPU driver (libze_intel_gpu.so) lives in the
      # `drivers` OUTPUT of intel-compute-runtime, not the default `out`.
      l0Driver = pkgs.intel-compute-runtime.drivers;
      # Runtime libs the L0 loader + Intel driver + torch wheel need at runtime.
      runtimeLibs = with pkgs; [
        level-zero # libze_loader.so — the Level Zero loader
        l0Driver # libze_intel_gpu.so.1 — the Intel L0 GPU driver (NEO)
        intel-compute-runtime # ocloc / OpenCL ICD
        ocl-icd
        zlib
        stdenv.cc.cc.lib # libstdc++ / libgcc_s for the wheels
        libxml2
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages =
          with pkgs;
          [
            uv
            python312
          ]
          ++ runtimeLibs;
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
          # Help the Level Zero loader find the Intel GPU driver explicitly.
          export ZE_ENABLE_ALT_DRIVERS="${l0Driver}/lib/libze_intel_gpu.so.1"
          # OpenCL ICD discovery for OpenVINO's intel_gpu plugin — point the
          # ocl-icd loader at intel-compute-runtime's vendor file (libigdrcl.so).
          export OCL_ICD_VENDORS="${pkgs.intel-compute-runtime}/etc/OpenCL/vendors"
          export NEOReadDebugKeys=1
          export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
          echo "xpu-bench shell: level-zero=$(echo ${pkgs.level-zero}) intel-compute-runtime=$(echo ${pkgs.intel-compute-runtime})"
          echo "LD_LIBRARY_PATH set; use: uv venv && uv pip install --index-url https://download.pytorch.org/whl/xpu torch && uv pip install sentence-transformers"
        '';
      };
    };
}
