cteate packages folder inside that create numpy_candles inside this package create a load_numpy_candles_from_binance_file that gets file path or url and load all columns into numpy array and return that. the array type would be float64 equivalant to diuble that can hold anything.
also i need to export NC that us a symplenamespace instance that maps names to column index. the dot notation names would be ts, o, h, l, c, v, q, n, vwap, vb, vs for timestamp in milisec, open, high, low, close, volume, quote volume, number of trades, candle vwap, buyer aggressive volume, seller aggresive volume
vwap must be calculate fir each candle by deviding quiute volume to volume and vs must be calculated using v - vb
the sample file is in the repository with th name BTCUSDT-1m-2026-03.csv and column names are available at the first line.
vb in the column name is the taker_buy_volume
ignore the ignore column

there is a consideration for vwap calculation.there maybe some candles that they may have zero volume or zero quote in either case the vwap may become 0 or nan and the vwap chart becomes discreet. so from index 0 to the end if any candle has zero volume or zero quote the vwap must be copied from its previous candle. if that done sequentialy from left to right of array then it guaranties that all vwap remains in range

now i need a numpy_candle_test function in same package. it gets numpy candle array as input and do these tests
all numbers in all columns must be equal to greater than zero also all of them must be valid number.
ts col diff to previous must be equal in all cells. so it guaranties no missing candle.
all close and open price must be between high and low
for all candles with none zero volume and qoute the vwap must be equal to quiute / volume
for all candles vs + vb == v
all candles with n greater than zero the volume and quote must be greater than zero
this function prints any error that it find or just say all tests passed

in same package create function numpy_candles_info that gets numpy candle as parameter and print some info in single line .
the shape
the time frame in seconds
duration of all candles in minutes and days. it is diff of last and first candles timestamp. do not forget that timestamps are in miliseconds.
first and last candle date time string

also create another function numpy_candles_filter_date that gets numpy candles and optional start_date and optional end_date or optional_count that returns a filtered candles between optional dates. dates are in datetime string format.

ensure that this package resides  in packages folder. we may add more packages later.
also create notebooks folder at the root of repository. under that create tests folder . under tests folder create numpy_candles_test.ipynb  then inside that load package and required other libraries and load candles , then run candle test function. then run candle info. then filter it to a single day like 2026-03-03 and again test that and print info of that. it must has exact 1440 members for a day.
then draw a ptice chart for vwap of that at the end.
and save the notebook.
then if there is any error please correct that  and finally save everything and commit the changes. 
if creating pr is mandatory please create and tell me what to do. if not please commit.
then tell me where can i run notebook that you created online.

if adding requirements.txt required let add that to the proper location but keep in mind that this is one of our notebooks. soon we will have tens of that

i saw the repo and found that nothing exists in main branch exept the sample data that I uploaded. so what is the procedure? does it like this or is there a need for creating pR ?

one question. how long is it reasonable to continue this session? is it good to continue that for full project ? or does it better to create new session for each project party ?

