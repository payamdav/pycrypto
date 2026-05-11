# Notebook for test of look back look ahead and normalization
>
* notebook file: /notebooks/tests/look_back_look_ahead.ipynb
* in a cell I want to set these parameters ( also they must have defaults as below):
    - asset: "BTCUSDT"
    - look_back: 1440
    - look_ahead: 240
    - datetime: "2025-12-12 20:00:00"
    - window_mode: "exclusive"
    - normalization mode: clip
    - k: 100 ( to keep 0.01 )

so what I want is to load desired candles with enough back and ahead candles and draw vwap + ohlc candles over datetime axis from look_back to current and also draw look_ahead candles in another chart side by side. so it is clear that left chart is past and right chart is upcoming. also keep in mind I want to have visually same measure on time axis on both charts because although the number of items is different in these two charts but I want the time axis give me same feeling so the upcoming would be smaller chart from x axis side.

also print all border line date times in that  or another cell . I mean current time that is in parameters. start and end of past candles. start and end of ahead candles.

then in another cell I want both vwap and time normalized as stated in instructions. for both past and ahead. and then again these two normalized series draws side by side. just like the above charts but is only vwap and do not have ohlc and their values normalized and clipped both for vwap and time.

>if any packages requires it must be install using %pip install at top of notebook
