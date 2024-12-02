Based on the CodeOcean repository from [this paper](https://ieeexplore.ieee.org/document/9118916/algorithms#algorithms)

My changes:
- "fix" a bunch of paths
    - the source code ocean assumes that this project is at the root of the filesystem?
      so it uses paths like '/data' instead of './data' Anyway, I think I changed
      all of those
- fix labels when training
    - for some reason it was passing a greyscale image as the label instead of
      the segmented one (see commit 6a4bcd64a8bb85f0e8af4644df79884b0ecbb498 "fix a couple other bugs" )
- add logs folder
- copy images before sending them to avoid some weird negative slicing error
- suppress all warnings (at the top of main.py) for clarity
    - you can unsupress these later if needed
- fix command line argument `--use_gpu` to work as expected


