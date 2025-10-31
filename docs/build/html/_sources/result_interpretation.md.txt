# Result Interpretation

TiRank outputs results in the `savePath/3_Analysis/` directory. Key outputs include:

* **📂 `spot_predict_score.csv`**:
    * Contains TiRank predictions in the `Rank_Label` column.

### Modes of Analysis

* **📉 Cox Survival Analysis**:
    * `TiRank+` spots indicate **worse survival outcomes**.
    * `TiRank-` spots indicate **better survival outcomes**.

* **✅ Classification**:
    * `TiRank+` spots are associated with the phenotype encoded as `1`.
    * `TiRank-` spots are associated with the phenotype encoded as `0`.

* **📈 Regression**:
    * `TiRank+` spots correspond to **high phenotype scores**.
    * `TiRank-` spots correspond to **low phenotype scores** (e.g., drug sensitivity).