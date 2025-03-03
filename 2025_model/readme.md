This readme summarizes the work that was done in 2025:

1. **Odds Model for NCAAW**
I tried to apply the mens odds model on the NCAAW data, but found that with the limited data I have there isn't evidence that it improves performance

2. **Injury Feature**
I built some scrapers to get both the NCAAM players data from torvik AND box scores from ESPN on the first game of the tourney.

Then I wrote some code to evaluate whether each player for each season/team from torvik actually played in the first game.

Naturally, there's some leakage here because we wouldn't know for sure that the player will play going forward.

I then created the feature in two ways:
* First calculate (Torvik Min% * PRPG!) / sum((Torvik Min% * PRPG!)) to get a player_perc_quality. Then calculate the sum of player_perc_quality among players who actually played in game 1 / the total player_perc_quality
* Second get Sum([avaialbility flag * PRPG!]) / Sum([% games played * PRPG!]) -- bascially this gives us the ratio of quality available now vs. during the season

My initial findings were that feature 1 worked better, but that it had a clear positive impact

3. **Player Stat Features**
I then started exploring deriving features directly from the torvik player data.

For all of the torvik stats, I calculated a variety of different metrics (mean, max, cv, gini, etc.) over top8, top5, and top3 players.

I then ran a large test introducing each of these new features (paired since its for t1 and t2) to the baseline model and evaluating performance.

I then introduced the best features one at a time and this resulted in the best model yet. 

However, I have a concern that the torvik player data could potentially incorporate the entire season team quality, which would mean they are leaking information. 

If that concern is unfounded, then I've made the most significant improvement to my model, potentially ever.

Baseline (Rolling CV 2013+): 0.186615
New (Rolling CV 2013+): 0.172781
