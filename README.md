# tsflex - feature-extraction benchmarking

## <p align="center"> <a href="https://predict-idlab.github.io/tsflex/#benchmark"><img alt="tsflex" src="https://raw.githubusercontent.com/predict-idlab/tsflex/main/docs/_static/logo.png" height="100"></a></p>

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?)](http://makeapullrequest.com) 

This repository withholds the [benchmark results](https://predict-idlab.github.io/tsflex/#benchmark) and visualization code of the `tsflex` paper and [toolkit](https://github.com/predict-idlab/tsflex).

## Flow

The benchmark process follows these steps for each feature-extraction configuration:

1. The corresponding feature-extraction Python script is called. This is done 20 times to average out the memory usage and create upper memory bounds. Remark that by (re)calling the script sequentially, no caching or memory is shared among the separate script-executions.
2. In this script:
   1. Load the data and store as a [pd.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
   2. [VizTracer](https://github.com/gaogaotiantian/viztracer) starts logging
   3. Create the feature extraction configuration
   4. Extract & store the features
   5. VizTracer stops logging
   6. Write the VizTracer results to a JSON-file

The existing [benchmark JSONS](code/benchmark_jsons/) were collected on a desktop with an *Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz* CPU and *SAMSUNG M393B1G73QH0-CMA DDR3 1600MT/s* RAM, with *Ubuntu 18.04.5 LTS x86\_64* as operating system.


## Instructions

To install the required dependencies, just run:

```bash
pip install -r requirements.txt
```

If you want to **re-run the benchmarks**, use the [run_scripts](code/run_scripts.ipynb) notebook to generate new benchmark JSONs and then visualize them with the [benchmark visualization](code/benchmark_visualizations.ipynb) notebook.


> We are open to new-benchmark use-cases via **pull-requests**!<br>
> Examples of other interesting benchmarks are different sample rates, other feature extraction functions, other data properties, ...


## Referencing our package

If you use `tsflex` in a scientific publication, we would highly appreciate citing us as:

```bibtex
@article{vanderdonckt2021tsflex,
    author = {Van Der Donckt, Jonas and Van Der Donckt, Jeroen and Deprost, Emiel and Van Hoecke, Sofie},
    title = {tsflex: flexible time series processing \& feature extraction},
    journal = {SoftwareX},
    year = {2021},
    url = {https://github.com/predict-idlab/tsflex},
    publisher={Elsevier}
}
```

---
<p align="center">
👤 <i>Jonas Van Der Donckt, Jeroen Van Der Donckt</i>
</p>
