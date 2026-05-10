# Normalization Hard Clipping based on last price
prepare "idea_normalize_based_on_last_price_clip.md" file inside /agents/ideas/  so whenever i mentioned about normalize_based_on_last_price_clip everything must be cleared for AI programming Agents.

> when inside a look_back_look_ahead window I stated that "price normalized based on last price, scaled by k and hard clip to -1 and 1 "
I mean that all prices ( or let say vwap) should be scaled using this formula scaled = k * (price - price_l) / price_l
where is vwap or price of each candle in the window and the price_l is the last price or vwap of last candle
then it must be clipped to the range of -1 to 1 

### Clipping mode ( default is hard)
* hard: as above, I mean that everything that is more than 1 become 1 and lower than -1 becomes -1
* tanh: use tanh for that

### clipping range 
> between -1 to 1 unless stated otherwise

### look back window and look ahead window
> normally this normalization targeted look back window or the one that uses for feature generation but in case that look ahead window mentioned for these normalization everything is like above expect that the base price would be the same price_l or the latest price in look back window

note that for both look back and look ahead window the base price or price_l is the right most candle in the look back window or vwap of last candle. and note that each changes window changes from minute to minute.

### k 
the k scale value could be stated or maybe asked to calculate. for example I may ask I want to keep ratio of 0.2 it means that after calculating (price - price_l) / price_l the k must be defined in a way that all 0.2 becomes 1 (k=5) so more than that will be clipped

please prepare the "idea_normalize_based_on_last_price_clip.md"

