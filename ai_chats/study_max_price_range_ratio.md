# Task: Study Max Price Range Ratio for Assets

### running environment for this task: gupyter notebook
> inside notebook to access huggingface dataset you can use huggingface_hub python library
I want the query to be run on duckdb ( provided by huggingface for dataset)  and just the result display in notebook

### Dataset information
> huggingface dataset: "payamdavaee/candles"
there are 7 assets there. each in its sub folder. I need the report for each of them separately
note: dataset is public . no need for credentials

### File structure
> Files of this task placed in /notebooks/studies/study_max_price_range_ratio.ipynb

### description
> 
* I need to look at the vwap column sorted by ts over all candles on a asset sub folder
* I want to have moving window with width 1440 samples preceding the current . all full width and complete windows.
* what I want is Max , Min, Right(current) of vwap inside window. so I want ABS((Max - Current) / Current) as max_up and ABS((Min - Current) / Current) as max_down then I want the bigger I mean maximum(max_up, max_down) for each window
* finally I want Deciles of this series 
