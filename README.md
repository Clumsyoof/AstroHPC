<a id="readme-top"></a>







<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#build">Build</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
     </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
AstroHPC is a astrophysics partical simulator made to be run on HPC clusters
<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

  #### 1. Core Toolchain & Compilers

  + CMake (version 3.20+)
  + GCC (version 9.0+) or Clang with C++17 support
  + Go (Version 1.21+)

  #### 2. Optional / Utilities

  + Python 3.8+ -> Only needed if you run the scripts to fetch fresh
  NASA JPL Horizons ephemerides or generate synthetic star clusters
  (scripts/generate_datasets.py)
  + Raylib 5.0+ (for 3D visualizer)

  #### 3. Terminal Requirements

  + A terminal emulator with UTF-8 and 256-color / Truecolor support
  (e.g. Alacritty, Kitty, WezTerm, GNOME Terminal, Windows Terminal)
   
  ### Installation

  #### Gentoo Linux
  ```sh
  emerge --ask dev-build/cmake sys-devel/gcc sys-cluster/openmpi dev-lang/go media-libs/raylib
  ```
  #### Ubuntu / Debian
  ```sh
  sudo apt update && sudo apt install build-essential cmake gcc g++ libopenmpi-dev openmpi-bin libomp-dev golang libgl1-mesa-dev libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libxi-dev
  ```
  #### Fedora / RHEL
  ```sh
  sudo dnf install cmake gcc gcc-c++ openmpi openmpi-devel libomp-devel golang raylib-devel
  ```

  ### Build
  ```sh
  # Standard CPU build (OpenMP intra-node multi-threading + MPI distributed memory enabled by default)
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)

  # Optional CUDA GPU backend build
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
  cmake --build build -j$(nproc)
  ```

<!-- USAGE EXAMPLES -->
## Usage

> [!NOTE]
> **HPC Parallelization Architecture**:
> - **MPI Domain Decomposition**: Particles are distributed along a 64-bit Morton space-filling curve. As particles drift across spatial domain boundaries, they are dynamically migrated between ranks via `MPI_Alltoallv`.
> - **Locally Essential Tree (LET)**: Rather than reducing remote ranks to a single point mass, each rank extracts a coarse subtree cut (down to depth 2, up to 64 sub-cells with exact centers of mass) exchanged via `MPI_Allgatherv`. Local particles evaluate remote forces against this essential tree.
> - **OpenMP Intra-Node Multi-Threading**: Traversal stacks are privatized per thread (`#pragma omp parallel`), parallelizing tree walk, direct $O(N^2)$ all-pairs, symplectic Leapfrog integration, and physical energy/momentum diagnostics.
> - **CUDA Backend**: Uses persistent device memory buffers across timesteps to eliminate allocation overhead, running a shared-memory tiled $O(N^2)$ direct offload kernel. MAC $\theta$ is bypassed on GPU runs since tree construction remains on CPU.

### Comprehensive HPC Test Suite
```sh
ctest --test-dir build --output-on-failure
```

### Hybrid Distributed Simulation (MPI + OpenMP)
```sh
# 4 MPI ranks x 8 OpenMP threads per rank = 32 compute cores
export OMP_NUM_THREADS=8
mpirun -np 4 ./build/nbody_sim -n 32768 -s 200 -b

# Distributed baseline accuracy comparison on Step 0:
# Evaluates distributed LET against exact global all-pairs direct force across all ranks
mpirun -np 4 ./build/nbody_sim -n 4096 --compare
```

### Single-Node Headless Simulator
```sh
./build/nbody_sim -n 8192 -s 100 -b   # Benchmark with phase timing breakdown
./build/nbody_sim -n 4096 --compare   # Compare single-node Barnes-Hut vs Direct O(N^2)
```

### Interactive TUI
```sh
./build/astro_tui                     # Default (4,096 bodies, galaxy disk)
./build/astro_tui -n 8192             # Custom particle count
./build/astro_tui -preset three_body  # 3-body orbital preset
```
*Controls: `[Space]` Pause/Resume &bull; `[R]` Reset &bull; `[Q]` Quit*

### 3D Visualizer
```sh
./build/astro_view                    # Live simulation
./build/astro_view <dataset.csv>      # Real star cluster (e.g. data/gaia_dr3_pleiades.csv)
./build/astro_view <snapshot_dir>     # Replay snapshots
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Advik.k -[LinkedIn](https://www.linkedin.com/in/advikkaushik06/) - advikkaushik478[at]gmail[dot]com
Project Link: [https://github.com/Clumsyoof/AstroHPC](https://github.com/Clumsyoof/AstroHPC)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



