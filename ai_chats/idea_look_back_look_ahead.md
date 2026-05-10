# Idea look back look ahead
> I have an idea that I want to run multiple studies and tests based on that. as some operation and preparation of data is same in all studies I want to describe the overall Idea and you prepare "idea_look_back_1ook_ahead.md" file inside /agents/ideas/  so whenever i mentioned about look_back_look_ahead everything must be cleared for AI programming Agents.
Suppose that we have an asset so we can load its candles data from start to end date
then I must define two parameters: look_back and look_ahead, these parameters are integers and define that how many preceding candles must be look to prepare features and how many ahead candles must be look at for creating labels ( in term of AI prediction and machine learning). if I omit the time frame the default is one minute.  so if I mention that look_back = 1440 and look_ahead = 240 I mean that for each observation I need to have 1440 preceding candles ( including last or current candle) or in other word I mean that the last or current candle plus 1339 preceding candles and  240 of next candles.
I may or may not mention a list of columns of candle that I may need. if not mentioned then full columns must be prepared.
some conventions:
* last_candle: the 1440th loaded candle or the latest in look back set
* last_time: last candles time (note that candles time are their open time, and we must note that all candles are finished.)
* current_time: last_time + time frame ( because we are now at closing of last candle)
* price: normally I mean vwap of candle
* current_price: last candles close price

## windowing
> the way that we observe on multiple or moving windows is as follows:
* loop: so there must be a loop or iterator from the datetime that it has 1440 look back items up to the datetime that it must have 240 look ahead items. I always tell the dates and if I say inclusive I mean the safe indexes that must be used is [1440:-240] of the date range that I mentioned. but if I say that exclusive I mean that I need all observation of my mentioned date range so for look back and look ahead data that it may need the program must load larger date boundaries.
* vectorized: in this case I mean that I do not want the loop but I need a 2d array for feature generation with the shape of (items, 1440) and a 2d array for label generation with the shape of (items, 24) , the concept of inclusive and exclusive remains same.  
the way that the vectorize must done is to use sliding_window_view from numpy.lib.stride_tricks or its equivalent method if other libraries other than numpy used.
* chunked vectorized: the vectorized model is fast but memory hungry. when I mentioned chunked vectorized I mean that there must be a loop that at each loop a number of N items vectorized or the 2D arrays become (N, 1440) and (N, 240) so the memory usage controlled.

### Historic indicator
> I may need to have some historic indicators to normalize or use as a reference while observing, so another date or index boundary requirement may be needed for that. for example I may say that for each observation I need 10 days moving average of volume or vwap. so the agent must consider to have a boundary of 1440 * 10 minutes for this indicator and it must be considered in addition to look back range. note that the concept of inclusive and exclusive are also valid here.
> note that when I silent about inclusive or exclusive the default is exclusive. so the Agent must prepare the script in a way that it loads more data in addition to my date range, so I have all required data in full timeframes between my date range.

now please prepare "idea_look_back_1ook_ahead.md" 
note that the user of this file is AI programming Agents.
