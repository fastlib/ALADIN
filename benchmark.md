## Reproduce benchmark

Requires [Docker](https://www.docker.com/) to run. A GPU is highly advised for faster execution, but a CPU works fine as well.

Reproducing the benchmark is only possible with a Hugging Face token granted by the author, as the iRhythm test dataset used was once public but is now unavailable. As such, the benchmark downloads a self-hosted copy of this testdataset.

Run the following from the repository ROOT to start the reproduction of the benchmark results on the iRhythm and CinC datasets, including the competitor results:
```bash
./benchmark/reproduce.sh
```

### Expected output
- JSON files with detailed metrics per run in the `results` folder
- Rendered figures in the `results` folder

### Duration
- ~1 hour on a GPU
- ~3 hours on a CPU
