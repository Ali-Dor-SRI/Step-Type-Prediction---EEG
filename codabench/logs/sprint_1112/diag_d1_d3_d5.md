## D1 — error anatomy of the slot-2 ensemble (w 0.3; pooled balanced accuracy per group)

### lopo: cell means X4f 0.6275, nets 0.5312, ens 0.6444

| runtype   |    n |    x4f |    net |    ens |
|:----------|-----:|-------:|-------:|-------:|
| calib     |  963 | 0.6532 | 0.5151 | 0.6511 |
| online    | 1692 | 0.6117 | 0.5408 | 0.6395 | 

| interface   |    n |    x4f |    net |    ens |
|:------------|-----:|-------:|-------:|-------:|
| BrainHero   | 1311 | 0.6232 | 0.5256 | 0.6461 |
| Graz        | 1344 | 0.6302 | 0.5372 | 0.6414 | 

| runtype   | interface   |   n |    x4f |    net |    ens |
|:----------|:------------|----:|-------:|-------:|-------:|
| calib     | BrainHero   | 459 | 0.6382 | 0.5164 | 0.6469 |
| calib     | Graz        | 504 | 0.6667 | 0.5139 | 0.6548 |
| online    | BrainHero   | 852 | 0.615  | 0.5305 | 0.6455 |
| online    | Graz        | 840 | 0.6083 | 0.5512 | 0.6333 | 

| posbin   |    n |    x4f |    net |    ens |
|:---------|-----:|-------:|-------:|-------:|
| 1-35     |  490 | 0.6243 | 0.5018 | 0.6424 |
| 36-70    |  490 | 0.6757 | 0.5225 | 0.6572 |
| 71+      | 1675 | 0.6132 | 0.5426 | 0.64   | 

|   run |   n |    x4f |    net |    ens |
|------:|----:|-------:|-------:|-------:|
|     1 | 468 | 0.6217 | 0.5064 | 0.6474 |
|     2 | 495 | 0.6828 | 0.5232 | 0.6545 |
|     3 | 420 | 0.619  | 0.5357 | 0.6286 |
|     4 | 420 | 0.6095 | 0.531  | 0.6238 |
|     5 | 426 | 0.5986 | 0.5352 | 0.6479 |
|     6 | 426 | 0.6197 | 0.561  | 0.6573 | 

| person   |   n |    x4f |    net |    ens |
|:---------|----:|-------:|-------:|-------:|
| P01      | 190 | 0.6949 | 0.5694 | 0.695  |
| P02      | 189 | 0.7189 | 0.5306 | 0.7388 |
| P03      | 189 | 0.525  | 0.5346 | 0.5407 |
| P04      | 566 | 0.5224 | 0.4411 | 0.5346 |
| P05      | 190 | 0.5475 | 0.4783 | 0.5526 |
| P06      | 187 | 0.6956 | 0.6105 | 0.7172 |
| P07      | 189 | 0.756  | 0.6348 | 0.777  |
| P08      | 189 | 0.7882 | 0.6909 | 0.8154 |
| P09      | 189 | 0.7149 | 0.488  | 0.7123 |
| P10      | 577 | 0.5897 | 0.5301 | 0.6215 | 

confusion (rows = true 0 mi / 1 sub / 2 word, row-normalised):

|   y |     0 |     1 |     2 |
|----:|------:|------:|------:|
|   0 | 0.627 | 0.169 | 0.203 |
|   1 | 0.13  | 0.657 | 0.213 |
|   2 | 0.14  | 0.213 | 0.647 | 

### later: cell means X4f 0.6735, nets 0.5803, ens 0.7013

| runtype   |   n |    x4f |    net |    ens |
|:----------|----:|-------:|-------:|-------:|
| calib     | 270 | 0.6969 | 0.5892 | 0.7161 |
| online    | 492 | 0.6606 | 0.5772 | 0.6931 | 

| interface   |   n |    x4f |    net |    ens |
|:------------|----:|-------:|-------:|-------:|
| BrainHero   | 378 | 0.7067 | 0.606  | 0.7541 |
| Graz        | 384 | 0.6406 | 0.5573 | 0.6484 | 

| runtype   | interface   |   n |    x4f |    net |    ens |
|:----------|:------------|----:|-------:|-------:|-------:|
| calib     | BrainHero   | 126 | 0.7297 | 0.6035 | 0.7476 |
| calib     | Graz        | 144 | 0.6667 | 0.5764 | 0.6875 |
| online    | BrainHero   | 252 | 0.6944 | 0.6071 | 0.7579 |
| online    | Graz        | 240 | 0.625  | 0.5458 | 0.625  | 

| posbin   |   n |    x4f |    net |    ens |
|:---------|----:|-------:|-------:|-------:|
| 1-35     | 140 | 0.7019 | 0.6087 | 0.7321 |
| 36-70    | 140 | 0.6935 | 0.5506 | 0.694  |
| 71+      | 482 | 0.6595 | 0.5825 | 0.6947 | 

|   run |   n |    x4f |    net |    ens |
|------:|----:|-------:|-------:|-------:|
|     1 | 131 | 0.7035 | 0.6112 | 0.7349 |
|     2 | 139 | 0.6909 | 0.5685 | 0.6986 |
|     3 | 120 | 0.6583 | 0.575  | 0.7    |
|     4 | 120 | 0.65   | 0.5833 | 0.6417 |
|     5 | 126 | 0.6508 | 0.619  | 0.7063 |
|     6 | 126 | 0.6825 | 0.5317 | 0.7222 | 

| person   |   n |    x4f |    net |    ens |
|:---------|----:|-------:|-------:|-------:|
| P04      | 377 | 0.5702 | 0.4606 | 0.6044 |
| P10      | 385 | 0.7748 | 0.7007 | 0.7962 | 

confusion (rows = true 0 mi / 1 sub / 2 word, row-normalised):

|   y |     0 |     1 |     2 |
|----:|------:|------:|------:|
|   0 | 0.602 | 0.16  | 0.238 |
|   1 | 0.115 | 0.7   | 0.186 |
|   2 | 0.111 | 0.087 | 0.802 | 

## D3 — label sequence structure (all 2,655 windows; measure only, feeds the flagged track)

class counts per run (mi, sub, word) -> number of runs:

| label                                      |   runs |
|:-------------------------------------------|-------:|
| (np.int64(10), np.int64(10), np.int64(10)) |     54 |
| (np.int64(12), np.int64(12), np.int64(12)) |     17 |
| (np.int64(12), np.int64(11), np.int64(10)) |      2 |
| (np.int64(11), np.int64(12), np.int64(11)) |      1 |
| (np.int64(11), np.int64(10), np.int64(12)) |      1 |
| (np.int64(10), np.int64(11), np.int64(12)) |      1 |
| (np.int64(12), np.int64(10), np.int64(11)) |      1 |
| (np.int64(10), np.int64(11), np.int64(11)) |      1 |
| (np.int64(11), np.int64(11), np.int64(12)) |      1 |
| (np.int64(10), np.int64(11), np.int64(10)) |      1 |
| (np.int64(11), np.int64(12), np.int64(10)) |      1 |
| (np.int64(9), np.int64(12), np.int64(12))  |      1 |
| (np.int64(11), np.int64(10), np.int64(9))  |      1 |
| (np.int64(11), np.int64(10), np.int64(10)) |      1 | 

transition matrix P(next | current), all runs:

|      |    mi |   sub |   word |
|:-----|------:|------:|-------:|
| mi   | 0.294 | 0.338 |  0.368 |
| sub  | 0.389 | 0.286 |  0.324 |
| word | 0.322 | 0.388 |  0.29  | 

same-class repeat rate 0.290 (iid balanced: 0.333); longest repeat per run: median 3, max 7

is the cue order shared? distinct label sequences per run number:

|   run | sequences         |
|------:|:------------------|
|     1 | 14 distinct of 14 |
|     2 | 14 distinct of 14 |
|     3 | 14 distinct of 14 |
|     4 | 14 distinct of 14 |
|     5 | 14 distinct of 14 |
|     6 | 14 distinct of 14 | 

|count(first half) - count(second half)| summed over classes, per run: median 5, max 16 (block-balanced design would give ≈ 0)

distinct label sequences per task (across person × session):

| task            | sequences         |
|:----------------|:------------------|
| AcquisitionBH   | 14 distinct of 14 |
| AcquisitionGraz | 14 distinct of 14 |
| OnlineRawBH     | 14 distinct of 14 |
| OnlineRawGraz   | 14 distinct of 14 | 

## D5 — what would explain sealed 0.736 vs lopo 0.644 / later 0.701 (slot-2 ens)

- lopo: cell means restricted to calib 0.6528, online 0.6397, windows 71+ 0.6396
- later: cell means restricted to calib 0.7198, online 0.6896, windows 71+ 0.6908

