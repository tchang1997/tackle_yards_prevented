# Assessing the value of tackles

We've all seen it before -- those touchdown-saving tackles, sacks on 3rd and long, or the running back skirting past the last man down the sidelines. But tackles -- or missed tackles -- occur on every play, and it's not clear how meaningful many of these tackles are.

This project aims to answer the question: "What if he missed the tackle?" Using tools from causal effect estimation, we propose the **Expected Yards Prevented (EYP)** tackle valuation metric: the number of yards that a ball carrier would have gained if a tackle was missed. Conversely, the **Expected Excess Yards Allowed (EEYA)** metric evaluates the number of yards that a tackler would have prevented if a tackle was completed.

## Setup

The basic formulation of tackle valuation as a causal problem is relatively simple: we are simply trying to estimate the conditional average treatment effect (CATE) of a successful tackle on yards after contact. Formally, we define treatment as a binary variable, which is 0 if a contact with a ball carrier resulted in an unsuccessful tackle, and 1 if contact resulted in a successful tackle (including forced fumbles -- *i.e.*, anything that stops the offense from progressing). For simplicity, we only consider the first instance of contact in each play. The outcome is the number of yards after contact.

## Challenges

Of course, successful tackles will reduce the expected yards after contact, and vice versa. But two questions remain: 1) *by how much?* and 2) *what factors about the play might change the effect?* In the language of causal inference, we call factors that could "explain" or influence the yards after contact besides the success of the tackle itself "confounders." If a QB is being blitzed, but there's a man open downfield, a sack would probably prevent an explosive completion. We might even take a player-based approach: Derrick Henry is probably much harder to bring down in the open field than your average RB, so the effect of missing a tackle is probably magnified. All this is to say that there are a myriad of player- and play-dependent factors that could explain the effect of a tackle (or a missed tackle) on yards after contact.

The challenge is controlling for these additional factors to isolate just the effect of a tackle vs. a missed tackle. We would like to answer this question: if everything about the play (e.g., the players themselves, their positions, formation, and everything else) was identical *except for* whether the first instance of contact resulted in a successful tackle, what would be the difference in outcome between these plays?

In a perfect world, we'd have both teams run two identical plays, and in one play, manipulate the universe such that the first contact results in a successful tackle. In the other play, we'd intervene such that the first contact results in a missed tackle. This setup is similar to a *randomized controlled trial* for testing the effect of medications: we have a treatment and a control group, where members of one group receive the drug, but members in the other group do not. But in football, this setup is impossible--it's vanishingly improbable that two plays are *exactly* the same, and furthermore, it's not clear how we'd "change the universe" to ensure that a tackle is successful in one play but not in another.

**Thankfully, the field of causal inference has been dealing with this problem for decades.** In practice, a lot of real-world interventions aren't randomized controlled trials--for example, rolling out a new law or policy and reasoning about if it "worked." But if we're willing to believe a few assumptions [CITE] about the data, we can still get an estimate of what would have happened if a tackle was missed (or not)--even though we don't explicitly observe that outcome.

We use a customized version of DragonNet [CITE], which jointly learns propensity scores and effect estimates for each treatment arm (three outputs) from a shared representation. We use a transformer-based encoder-only model that aggregates the time-varying tracking features with play-level features to create this shared representation.

## Modeling

Our outcome and propensity models take in essentially the same covariates, except that the outcome model additionally takes into account whether or not an attempted tackle was successful. As features, inspired by [CITE], we take in (for each play):
* Geometric features (relative distance, speed, and acceleration with respect to ball carrier)  using tracking data for all players up to the time of first contact
* Absolute tracking features using tracking data up to the time of first contact
* Player-level information for all players on the field (e.g., height, weight, age)
* Play/game-level information (e.g., team ID, field position, clock)

We base our models on a simplified version of the BERT transformer architecture often used in sentence classification settings. We hypothesize that its capabilities extend to general sequence classification.
