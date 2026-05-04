please inside the repository create a folder with the name of "scripts" inside that create another folder with name of candle_downloader inside this folder create a python file with name download_monthly_candle_file_and_upload_to_[huggingface.py](http://huggingface.py)
also create a configuration file that the name is iconfig.toml and this file must have a "settings" section and under that assets = ["BTCUSDT", "ETHUSDT", "TRUMPUSDT", "VINEUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]
this python script must be a github action and runs manually by me. so do what ever it requirds to be an action.
the goal of this script is as bellow:
connect to my huggingface dataset with repo id: "payamdavaee/candles" , inside that each asset must have a folder. if there is not please create a folder for each. then inside each folder get a list of all files. the file names are somthing like BTCUSDT-1m-2026-03.csv that says its asset name as BTCUSDT , 1m for one minute period, 2026 for year and 03 for month , then the script can understand which month is the last month that the monthly candle of that asset is available.
our policy is that to have all monthly candle files with one minute period from 2024-01 up to now. 
the files must be downloaded from Binance public historical data archive. the sample url of that is something like https://data.binance.vision/data/futures/um/monthly/klines/{asset_name}/1m/{asset_name}-1m-{year}-{month}.zip that all curly braces inside that must replaces with their real value.
keep in mind that binance will put the monthly candle files of each asset 2 days after the month finished. so we expect that if for example current date is 2026-05-02 or later the files for 2026-05 becomes available.  also please note that not all assets started from 2024-01 so some of them may return 404 error if trying to download files for previous months.
so everytime the script runs check for latest available file in huggingface dataset "payamdavaee/candles" under its folder. after that the script extract current year month. if it finds that new file is available it must download the file from binance. then decompress it and upload it to hugging face dataset. in other word everytime that the script runs it ensures that the current files in huggingface dataset is up to date.
it is clear that when it first runs it must download all files from 2024-01 of all assets from binance and upload them to huggingface. and if immediatly after that we run the script again it must do nothing because the script updated the dataset a few minutes ago and everything is up to date. 
I expect that the action reports something about number of files that it fetchs and stored in huggingface for each asset so i think that this report can be visible in action debug or logs.
to access huggingface dataset you can use huggingface_hub python library but do not forget to install that or include required files for actions. also to access dataset the program must login withj access token that is available as github secret token with the name of HUGGINGFACE_FULL_TOKEN


does this action depends on nodejs ? because there is a warning :
Node.js 20 actions are deprecated. The following actions are running on Node.js 20 and may not work as expected: actions/checkout@v4, actions/setup-python@v5. Actions will be forced to run with Node.js 24 by default starting June 2nd, 2026. Node.js 20 will be removed from the runner on September 16th, 2026. Please check if updated versions of these actions are available that support Node.js 24. To opt into Node.js 24 now, set the FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true environment variable on the runner or in your workflow file. Once Node.js 24 becomes the default, you can temporarily opt out by setting ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true. For more information see: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/



inside the repository folder "scripts" create another folder with name of depth_snapshot_mover ,  inside this folder create a python file with name depth_snapshot_mover_to_[huggingface.py](http://huggingface.py)
also create a configuration file that the name is iconfig.toml and this file must have a "settings" section and under that assets = ["BTCUSDT", "ETHUSDT", "TRUMPUSDT", "VINEUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]
this python script will be run on a vps with debian os and python 3.12
the goal of this script is as bellow:
connect to my huggingface dataset with repo id: "payamdavaee/depth_snapshot" , inside that each asset must have a folder. if there is not please create a folder for each.
inside vps there is a running process that periodically get depth snapshot of assets from binance and store it in /var/data/{asset in lowercase}  the filename can be either {asset in lowercase}_depth_snapshot_{timestamp in milisecond}.json  like btcusdt_depth_snapshot_1777725602137.json  or it could be the compressed one {asset in lowercase}_depth_snapshot_{timestamp in milisecond}.json,gz  like btcusdt_depth_snapshot_1777725602137.json.gz
our policy is to move all files stored in /var/data/{asset} to huggingface dataset payamdavaee/depth_snapshot under their relevant {asset} folder, and if the file compressed it must be decompressed before moving to huggingface dataset. and after successfull movement it must be deleted.
so everytime the script runs it looks inside each folder of assets in /var/data/{asset}  then it gets a list of all files inside there. then if some of files are compressed , decompress them. then upload them to huggingface payamdavaee/depth_snapshot/{asset}  then after successfull upload remove them from server.
be aware that currently more than 100000 files are in each asset folders so iot is better idea to fecth get list of files in batch of 100 
I expect that the script reports something about number of files that it fetchs and stored in huggingface for each asset.
to access huggingface dataset you can use huggingface_hub python library but do not forget to requirements.txt. also to access dataset the program must login with access token that is available in HUGGINGFACE_FULL_TOKEN environment variable
I worry that the bug in script causes deleting files while it is not uploaded correctly to dataset. for this reason I expect that at the first step for test purpose in each run the script only do the job for 1 asset and one batch of 100  files. and after that exit from script. also instead of deleting files just rename them and add ".del" postfix to them. but prepare the script in a way that if it passes this test just by zero effort it goes for multi asset and multi batch.
alsi it is good idea to have some feedback from script for example after each batch of 100 files says Asset: {asset} - batch: {batch number} - time duration: {3s} - batch remains: {1000}
please note that the asset names everywhere must be in lowercase while in iconfig.toml maybe lowercase or uppercase
when you finished let me know how I must run it on my debian vps



Task name: Create daily Parquet file from depth snapshot files
Folder name: "create_daily_depth_snapshot_parquet_files" under /scripts 
Script file name: create_daily_parquet_[files.py](http://files.py)
Running environment: on Linux VPS equipped with Python 3.12
Description: An external script takes snapshot of asset depth and save it as json file in current working directory. the file maybe compressed using gzip or may not. in case of compression the file extension is gz and it must decompressed to use. the file names are like {asset}_depth_snapshot_{ts}.json that asset is an asset name in lowercase  and ts is a 13 digit timestamp in milliseconds the example of file name is btcusdt_depth_snapshot_1777796102211.json for uncompressed and btcusdt_depth_snapshot_1777796102211.json.gz for compressed one.
all files in current working directory holds snapshot of single asset and files of different assets do not mixed in folders.
having timestamp in filenames make it able to look at filename in sorted manner.
what the script do: it generate a list of files in current working directory that is ending with json or json.gz , sort them and iterate over them one by one. 
from the first filename it must extract the asset name.
then if the file if compressed it must be uncompressed.
no need to uncompress the original file on disk . it must remain untouched. just uncompresse the contents. 
then json must be loaded. each file has these fields:
"lastUpdateId": ignore
"E": ignore
"T": 13 digit timestamp integer value
"bids": list of list ( double price stored as string , double volume stored as string) example [["78409.20","2.696"],["78409.10","0.008"]]
"asks": list of list ( double price stored as string , double volume stored as string) example [["78409.20","2.696"],["78409.10","0.008"]]
when first file evaluated at first step the timestamp presented as "T" must be examined to extract the date (year-month-day)
so at this step we understand that we  are creating parquet file for this date. so we iterate over files while stay at the same day . when ever we reach another day we ignore that file and we start building parquet file and writing that. then we add "del" extension to all files that processed and belongs to current day.
after that we go for next day and we start reading the last file that we ignored earlier and do what we did for next day.
like this we go to create parquet file for each day and add "del" extension to json files.
how to prepare daily parquet files: each daily parquet files build from multiple json files. the columns that must be stored in parquet files are as below:
"ts" type timestamp(ms) retrieved from "T" in json
"bids" type list_(level) from bids in json where level = struct([field("price", float64()), field("volume", float64())])
"asks" type list_(level) from asks in json where level = struct([field("price", float64()), field("volume", float64())])
there are some computed columns that I want to store. keep in mind that for bids and asks they become list of list of two strings that first string represent a double Price and the second string represent double volume. and also if we consider indexes of the bids and asks list we will see that bids sorted descending and ask sorted ascending so bids[0][0] is the greatest price in bids and asks[0][0] is the lowest price bids
computed columns:
"bcount": len(bids)
"acount": len(asks)
"bmin": b[-1][0]
"bmax": b[0][0]
"amin": a[0][0]
"amax": a[-1][0] 
"arange": amax - amin
"brange"" bmax - bmin
"spread": amin - bmax
"mid": (amin + bmax) / 2
"av": sum(a[:][1])  some of all asks volumes
"bv": sum(b[:][1])  some of all bids volumes

the parquet file must be named {asset}_depth_snapshot_{year}_{month}_{day}.parquet
after creating each file a single line report must printed that shows the file name including date, number of json files that used for that, (last ts - first ts) in minutes , spread and mid, av, bv, time taken to create file in seconds

for sake of test purpose I prefer have the ability to send number of days when running the script. so if i call it with 3 days then after creating 3 files it must exit from script. 
to create parquet files use pyarrow library 
do not forget to include requirements.txt file and as I must run the script on another vps at the end of job tell me the instruction how to run it on that vps


