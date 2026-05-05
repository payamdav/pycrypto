# Task: Study Max logarithmic Price Range Ratio for Assets

# description
> this task overally like the task Study Max Price Range Ratio for Assets with some changes and different outcome

### running environment for this task: gupyter notebook
> inside notebook to access huggingface dataset you can use huggingface_hub python library
I want the query to be run on duckdb ( provided by huggingface for dataset)  and just the result display in notebook

### Dataset information
> huggingface dataset: "payamdavaee/candles"
there are 7 assets there. each in its sub folder. I need the report for each of them separately
note: dataset is public . no need for credentials

### File structure
> Files of this task placed in /notebooks/studies/study_max_log_price_range_ratio.ipynb

### description
> 
* I need to look at the vwap column sorted by ts over all candles on a asset sub folder
* I want to have moving window with width 1440 samples preceding the current . all full width and complete windows.
* I want to find largest vwap that it hast most distant from right or current. so in each window I need to find the vwap that has max(abs(vwap - right)) lets call that max_vwap
* then I want to have ln(max_vwap / right) lets call that log_return
* so we have a log_return of each moving window ( that is the largest log return in that window)
* if we calculate the percentile of that I need to print these 
name of asset
{ percentile %0 to %100 in step of 10}
{ percentile %90 to %100 in step of 1}
histogram chart with 100 bins
a separator between this and next asset
