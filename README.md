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
  emerge --ask dev-build/cmake sys-devel/gcc dev-lang/go media-libs/raylib
  ```
  #### Ubuntu / Debian
  ```sh
  sudo apt update && sudo apt install build-essential cmake gcc g++ golang libgl1-mesa-dev libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libxi-dev
  ```
  #### Fedora / RHEL
  ```sh
  sudo dnf install cmake gcc gcc-c++ golang raylib-devel
  ```

  ### Build
  ```sh
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  ```

<!-- USAGE EXAMPLES -->
## Usage

> [!NOTE]
> The disk preset provides an idealized mock galactic disk (power-law surface density and Salpeter IMF) for visualization and benchmarking, not an exact self-consistent Jeans-theorem equilibrium.
> The CPU compute backend uses a dynamic Structure-of-Arrays (SoA) layout and 64-bit Morton curve decomposition. A modular CUDA backend interface is available for cluster builds (`-DENABLE_CUDA=ON`).

### Comprehensive HPC Test Suite
```sh
ctest --test-dir build --output-on-failure
```

### Interactive TUI
```sh
./build/astro_tui                     # Default (4,096 bodies, galaxy disk)
./build/astro_tui -n 8192             # Custom particle count
./build/astro_tui -preset three_body  # 3-body orbital preset
```
*Controls: `[Space]` Pause/Resume &bull; `[R]` Reset &bull; `[Q]` Quit*

### Headless Engine & Verification
```sh
./build/nbody_sim -n 8192 -s 100 -b   # Benchmark with phase timing breakdown
./build/nbody_sim -n 4096 --compare   # Compare Barnes-Hut vs Direct O(N^2)
```

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



