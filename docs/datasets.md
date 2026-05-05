# Datasets

Four public datasets ship with loaders. Two are auto-downloaded; the
other two require a free login on Ods.ai and a manual file placement.

## Hillstrom (MineThatData)

Kevin Hillstrom's classic e-mail marketing experiment. ~64k rows, three
treatment arms (no email / mens email / womens email), three outcomes
(visit, conversion, spend).

* **Auto-download.** `uv run uplift-bench download hillstrom`
* **URL.** `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`
  (HTTP only; HTTPS fallback via a GitHub mirror).
* **Default contrast.** Womens E-Mail vs No E-Mail — has the strongest
  measured ATE in published analyses.
* **Default outcome.** `visit` — `conversion` is too rare to give a
  meaningful Qini at this n.

## Criteo Uplift Prediction Dataset v2.1

The standard uplift benchmark. ~13.9M rows, 12 anonymised features,
binary treatment, two outcomes (visit, conversion).

* **Auto-download.** `uv run uplift-bench download criteo`
* **URL.** `https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz`
  (~297 MB compressed, ~1.5 GB uncompressed).
* **Cache.** First load decodes the CSV.gz and caches a parquet next to
  it; subsequent loads parse in seconds.
* **Subsample.** For local iteration use
  `dataset.loader_params.subsample=1_000_000` to grab a fixed-seed
  one-million-row slice.

## Lenta uplift dataset

Russian grocery loyalty campaign published by Lenta and packaged into
the scikit-uplift library.

* **Auto-download.** `uv run uplift-bench download lenta`
* **URL.** `https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz`
  (~138 MB compressed, ~567 MB uncompressed). Same source-of-truth URL
  that `sklift.datasets.fetch_lenta` uses, so this loader stays in
  lock-step with upstream without depending on it at runtime.
* **Schema.** ~687k rows, binary treatment (`group` = test/control,
  ratio ~0.75/0.25), binary outcome (`response_att`, ~10% positive).
  We retain ~14 most informative numeric features plus a one-hot
  `gender` block; the full ~190-column raw schema is documented at
  <https://www.uplift-modeling.com/en/v0.5.1/api/datasets/fetch_lenta.html>.
* **Cache.** First load decodes the CSV.gz and caches a parquet next to
  it; subsequent loads parse in seconds.
* **Sample fixture.** `data/sample/lenta/lenta_dataset.csv.gz` for
  offline tests / smoke runs.

## X5 RetailHero Uplift Modeling Contest

Russian retail-loyalty contest dataset. Login-walled on Ods.ai.

* **Manual download.** Sign in at
  <https://ods.ai/competitions/x5-retailhero-uplift-modeling/data>
  and place
  ```
  data/raw/retailhero/uplift_train.csv
  data/raw/retailhero/clients.csv
  ```
* **Loader.** `RetailHeroLoader` joins clients onto the uplift table,
  median-imputes missing numerics, one-hots gender.
* **Sample fixture.** A tiny synthetic stand-in lives at
  `data/sample/retailhero/` so CI smoke runs work offline.

## MegaFon Uplift Competition

Telco dataset, ~600k rows, 50 anonymised numeric features.

* **Manual download.** Sign in at
  <https://ods.ai/competitions/megafon-uplift-competition/data> and
  place
  ```
  data/raw/megafon/train.csv
  ```
* **Loader.** `MegaFonLoader` discovers `X_*` features at read time;
  normalises both treatment-column variants ("treatment_group" string and
  numeric).
* **Sample fixture.** `data/sample/megafon/`.

## Sample fixtures

Every loader can read from `data/sample/<dataset>/` instead of
`data/raw/<dataset>/`. Sample files match the production schemas but
contain only a few thousand synthetic rows. Useful for tests and
confirming the install without downloading anything.
