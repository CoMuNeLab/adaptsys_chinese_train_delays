# Code to run prepare the analysis

The code should be run in the following order:

```
.
├── job_fit_params.py
├── job_fit_params_regression.py
├── job_fit_params_validate.py
├── job_coverage.py
├── job_coverage_plot.py
├── job_delay_rain_corr.py
├── job_ext_field_response.py
├── job_ext_field_response_plot.py
├── job_steady_state.py
└── job_steady_state_plot.py
```


This will produce a `plots` folder with all the plots from the paper.

To run each of those scripts, you can call them with `uv run job_xxx.py`.
