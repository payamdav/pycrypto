# Task: Huggingface Parquet Candle Dataset
## create and maintain Huggingface crypto candle dataset in parquet format for multiple crypto assets in separate folders

### running environment for this task: github action
> creating github files for this project to be an action is part of task

### Dataset information
> huggingface dataset: "payamdavaee/candles"
each asset must have its own folder. if that is not exist it must created.
candle data stores in monthly files and their filename is in this pattern {asset}_1m_{year}_{month}.parquet where asset is in lowercase, the example file name is : btcusdt-1m-2026-03.parquet
to access huggingface dataset you can use huggingface_hub python library but do not forget to install that or include required files for actions. also to access dataset the program must login with access token that is available as github secret token with the name of HUGGINGFACE_FULL_TOKEN

### File structure
> Files of this task placed in /scripts/huggingface_parquet_candles folder
executable file: update_huggingface_parquet_candles.py
config file: config.toml

### Config file
> include a settings section that has a list of assets. ["BTCUSDT", "ETHUSDT", "TRUMPUSDT", "VINEUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]

### Candle files
> candle files retrieved from binance monthly csv candle files. the url of that is https://data.binance.vision/data/futures/um/monthly/klines/{asset_name}/1m/{asset_name}-1m-{year}-{month}.zip that all curly braces inside that must replaces with their real value. sample is https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-04.zip
then files must be decompress to get csv file.

### creating parquet candles
> candle files that downloaded and decompressed must passes to load_numpy_candles_from_binance_file() that reside in packages/numpy_candles in this repository then it returns a numpy array. next the parquet file must be created using pyArrow library
the parquet file has 11 columns as the numpy_candle has, all columns type are double except the ts column that its datatype is timestamp(ms) and n column that its type is int32 the column names are same as exposed by simplenamespace NS in that package. I mean ts, o, h, l, c, v, q, n, vwap, vb, vs.

### Parquet candles date range
> for each asset I expect that the monthly files become available from 2024-01 up to now. with two consideration:
1. some assets may not been available from 2024-01 and their starting date may been later than that.
2. binance may release monthly candles 2 days later than months end.

> in either case trying to retrieve the file from binance results in 404 error that must correctly handled by script.


### script procedure
> when it runs iterate through assets in config file then it checks availability of folder in huggingface dataset and if there is not it must be created. then it get files list and found the last month that the monthly candles available for that asset. then the script retrieves current month from datetime on the running environment. and it starts to download required files from binance and create numpy candle and then parquet file and then upload that to desired folder on the huggingface dataset.

### test run
> for test purpose I suggest that the running mechanism of this github actions become manual. and when it runs it just do the job for only one month. and create only next month file for each asset. then if everything tested and looks fine we can change the script to do whole job.
