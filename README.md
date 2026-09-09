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

  + GCC (version 9.0+) or Clang with OpenMP support
  + OpenMP Runtime: libgomp (included with GCC)
  + Go (Version 1.21+)
  + make

  #### 2. Optional / Utilities

  + Python 3.8+ -> Only needed if you run the scripts to fetch fresh
  NASA JPL Horizons ephemerides or generate synthetic star clusters
  (scripts/generate_datasets.py)

  #### 3. Terminal Requirements

  + A terminal emulator with UTF-8 and 256-color / Truecolor support
  (e.g. Alacritty, Kitty, WezTerm, GNOME Terminal, Windows Terminal)
    ──────
  ### Installation

  #### Gentoo Linux
  ```sh
  emerge --ask sys-devel/gcc sys-devel/make dev-lang/go
  ```
  #### Ubuntu / Debian
  ```sh
  sudo apt update && sudo apt install build-essential gcc make golang libomp-dev
  ```
  #### Fedora / RHEL
  ```sh
  sudo dnf install gcc make golang libgomp
  ```

  ### Build
  ```sh
  make
  ```
  * `./astro_tui` — Interactive Bubble Tea terminal visualizer
  * `./nbody_sim` — Headless C benchmark engine

<!-- USAGE EXAMPLES -->
## Usage

### Interactive TUI
```sh
./astro_tui                     # Default (4,096 bodies, galaxy disk)
./astro_tui -n 8192             # Custom particle count
./astro_tui -preset three_body  # 3-body orbital preset
```
*Controls: `[Space]` Pause/Resume &bull; `[R]` Reset &bull; `[Q]` Quit*

### Headless C Benchmark
```sh
./nbody_sim -n 8192 -s 100 -b   # Benchmark with phase timing breakdown
./nbody_sim -n 4096 --compare   # Compare Barnes-Hut vs Direct O(N^2)
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

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Advik.k -[LinkedIn](https://www.linkedin.com/in/advikkaushik06/) - advikkaushik478[at]gmail[dot]com
Project Link: [https://github.com/Clumsyoof/thread-picruncher](https://github.com/Clumsyoof/AstroHPC)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



