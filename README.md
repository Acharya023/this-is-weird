# This Is Weird

**This Is Weird** is a domain-agnostic anomaly-discovery system.

Its question is not:

> "What is popular?"

It is:

> **"What is changing unusually enough that it deserves investigation, and what evidence explains the change?"**

The system is designed to discover unexpected patterns across real-world data rather than start with a fixed list of topics.

## Core architecture

```
                    REAL-WORLD SIGNALS
                           |
          +----------------+----------------+
          |                |                |
       Structured        News            Social
         data             web          conversations
          |                |                |
          +----------------+----------------+
                           |
                    Observation layer
                           |
                  Anomaly / change detection
                           |
                Cross-signal relationship layer
                           |
                    Investigation
                           |
                  Evidence + NLP/ML
                           |
                 Human-readable discovery
```

The important separation is:

**Detection finds the unusual thing. Investigation tries to understand it.**

No language model should be allowed to manufacture a numerical anomaly. Likewise, statistical detection alone should not be expected to explain a human phenomenon.

## Domains

The architecture is intentionally open-ended.

Potential signal domains include:

- financial markets
- social-media conversations
- news and emerging topics
- internet culture / memes / "brainrot"
- movies and TV
- games
- music
- creators and celebrities
- technology and apps
- companies and brands
- products and consumer trends
- sports
- geopolitics and world events
- politics and elections, descriptively and without political persuasion
- previously unknown topics

These are **domains of observation, not a hard-coded topic whitelist**.

A newly emerging topic should be discoverable even if the system has never seen its name before.

## Social-media principle

Social signals are not intended to become a generic sentiment dashboard.

The system should look for changes such as:

- unusual mention velocity
- sudden topic emergence
- new phrases or entities
- unusual co-occurrence of entities
- discussion spreading between communities
- sentiment or emotional distribution changing unusually
- disagreement between social activity and other signals
- a topic becoming large before mainstream coverage
- a previously quiet topic suddenly returning

The same engine should be capable of finding an unexpected meme and an unexpected geopolitical discussion. Their importance is investigated after detection rather than assumed beforehand.

## Cross-signal discovery

The most interesting candidates may be relationships between domains:

```
Social spike
    ↓
News appears
    ↓
Market reaction
```

or:

```
Market anomaly
    ↓
No obvious news
    ↓
Unusual social discussion
```

or:

```
Social anomaly
    ↓
No known entity
    ↓
New topic discovered
```

Therefore the system should not score each domain independently and simply add the numbers. It should eventually reason about **agreement, disagreement, timing and propagation between signals**.

## Current implementation status

The repository began as a narrow Indian-market experiment. That work remains useful as the first domain adapter.

The current market detector has validated:

1. stock-specific return anomalies
2. unusual trading activity
3. cross-stock relationship/divergence signals
4. market-relative context

The current investigation layer has also validated:

- Google News discovery
- source-family normalization
- event grouping
- target-aware financial evidence extraction
- financial facts and quantities
- financial NER
- sentiment
- experimental relation extraction

Those components should now be treated as **market-domain components**, not as the definition of the whole project.

## Domain-neutral discovery core

`src/core/discovery.py` contains the shared representations:

- `Observation`
- `AnomalySignal`
- `DiscoveryCandidate`

A domain adapter should convert its raw data into these structures. This allows a market anomaly and a social-media anomaly to enter the same downstream discovery/investigation architecture without pretending their raw measurements are identical.

## Current market discovery architecture

The original 20-stock experiment is now explicitly a **controlled regression fixture**, not the discovery universe.

The market path is:

```
Hugging Face historical market data
              |
              v
     discover observed universe
              |
              v
       adjusted-price history
              |
              v
       population-wide features
              |
       +------+------+
       |             |
 historical      cross-sectional
 abnormality      abnormality
       |             |
       +------+-----+
              |
              v
      candidate discoveries
              |
              v
     peer / market context
              |
              v
       internet investigation
```

The population detector intentionally does **not** read `config/bootstrap_symbols.txt`. That file remains useful for small deterministic regression tests.

## Historical data strategy

We should not put a giant market-history file inside the Git repository.

**Hugging Face is the persistent data layer.**

The current market adapter uses the public `tejhq/indian-markets` dataset, which provides broad NSE history, including back-adjusted prices and raw volume. Runtime downloads are cached locally and the repository stores only code/configuration.

For data that **we derive ourselves** — normalized observations, population-level features, anomaly candidates, daily discovery scores, cross-signal relationships, and investigation results — we can maintain a separate project-owned Hugging Face dataset as the historical store. This lets the system accumulate history across runs without turning Git into a data warehouse.

**GitHub = code and reproducible logic. Hugging Face = large, persistent data/history.**

## Development strategy

We will expand in layers rather than prematurely build the entire internet.

### Stage 1 — Broaden the market observation space

Move beyond the experimental 20-stock universe toward:

- NIFTY 50 / broader equity universe
- sector indices
- market breadth
- volatility and other market context
- longer historical baselines

The existing market detector remains useful for validating the architecture.

### Stage 2 — Build the social observation layer

Start with reproducible public/authorized sources rather than scraping everything indiscriminately.

The social layer should produce time-windowed observations such as:

- entity/topic mention counts
- unique-post counts where available
- discussion velocity
- semantic clusters
- emerging phrases
- source/community distribution

### Stage 3 — Discover unknown topics

Use embeddings, clustering and temporal novelty detection to find topics without requiring a predefined taxonomy.

### Stage 4 — Cross-domain correlation

Connect market, news and social observations through:

- entities
- timestamps
- semantic similarity
- event relationships
- propagation patterns

### Stage 5 — Investigation

Only after a candidate survives detection should heavyweight NLP/search be used to answer:

- What happened?
- What entities are involved?
- What changed?
- When did it begin?
- Which sources independently report it?
- Which signals agree or conflict?
- What evidence supports the explanation?

## Hugging Face

Hugging Face models and datasets should provide genuine ML/data capabilities where they improve the system:

- embeddings
- semantic clustering
- entity extraction
- event extraction
- sentiment/emotion where useful
- domain-specific language understanding
- historical datasets

Models are components of the system, not decoration. A model should earn its place through a benchmark.

## Engineering rule

Every major new layer gets its own reproducible repository test before it is connected to production scoring.

The workflow is:

```
change repository
      ↓
add/update test script
      ↓
restart runtime
      ↓
git pull
      ↓
run test
      ↓
inspect results
      ↓
only then integrate
```

## Current principle

**Do not optimize the explanation engine before the discovery universe is broad enough.**

The original 20-stock experiment taught us how to build anomaly detection and evidence extraction. The next phase is to make the underlying discovery engine genuinely domain-agnostic.
