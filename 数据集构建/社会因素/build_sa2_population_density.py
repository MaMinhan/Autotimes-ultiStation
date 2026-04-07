import geopandas as gpd
import pandas as pd

# ============================
# 路径（改成你自己的）
# ============================
GPKG_PATH = r"C:\Users\maminhan\Desktop\AusgridSA2\ASGS 2016 Volume 1.gpkg"
POP_CSV  = r"C:\Users\maminhan\Desktop\AusgridSA2\2016 Census GCP Statistical Area 2 for NSW\2016Census_G01_NSW_SA2.csv"
OUT_CSV = r"C:\Users\maminhan\Desktop\AusgridSA2\out_pop_density\NSW_SA2_population_density.csv"
OUT_GEOJSON = r"C:\Users\maminhan\Desktop\AusgridSA2\out_pop_density\NSW_SA2_population_density.geojson"

# 如果你的 POP_CSV 只覆盖 NSW，就设 True（推荐）
FILTER_TO_NSW = True

# 读取 SA2 边界层（ASGS Volume 1 通常是这个 layer 名）
SA2_LAYER_NAME = "SA2_2016_AUST"

def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def main():
    print("▶ Loading SA2 geometry...")
    sa2 = gpd.read_file(GPKG_PATH, layer=SA2_LAYER_NAME)

    # 尝试定位 SA2 code/name/state 字段（不同版本字段可能略有差异）
    sa2_code_col = pick_col(sa2.columns, ["SA2_MAINCODE_2016", "SA2_CODE_2016"])
    sa2_name_col = pick_col(sa2.columns, ["SA2_NAME_2016", "SA2_NAME"])
    state_col    = pick_col(sa2.columns, ["STATE_NAME_2016", "STATE_NAME", "STE_NAME_2016"])

    if not sa2_code_col or not sa2_name_col:
        raise RuntimeError(f"SA2 layer missing key cols. Found code={sa2_code_col}, name={sa2_name_col}. "
                           f"Available cols: {list(sa2.columns)}")

    keep_cols = [sa2_code_col, sa2_name_col, "geometry"]
    if state_col:
        keep_cols.insert(2, state_col)
    sa2 = sa2[keep_cols].rename(columns={sa2_code_col: "SA2_CODE", sa2_name_col: "SA2_NAME"})
    if state_col:
        sa2 = sa2.rename(columns={state_col: "STATE_NAME"})

    # 可选：只保留 NSW（当人口表只有 NSW 时强烈建议）
    if FILTER_TO_NSW:
        if "STATE_NAME" not in sa2.columns:
            print("⚠️ SA2 layer has no STATE_NAME column; cannot filter NSW reliably. "
                  "You should either (1) add state lookup, or (2) use Australia-wide population CSV.")
        else:
            sa2 = sa2[sa2["STATE_NAME"].astype(str).str.lower().eq("new south wales")].copy()

    # 计算面积（km²）
    sa2 = sa2.to_crs(epsg=3577)  # 澳大利亚 Albers（适合算面积）
    sa2["area_sqkm"] = sa2.geometry.area / 1e6

    # SA2_CODE 统一成字符串（避免 0 前导/类型不一致）
    sa2["SA2_CODE"] = sa2["SA2_CODE"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(9)

    print("▶ Loading population CSV...")
    pop = pd.read_csv(POP_CSV)

    pop_code_col = pick_col(pop.columns, ["SA2_CODE_2016", "SA2_MAINCODE_2016", "SA2_CODE"])
    if not pop_code_col:
        raise RuntimeError(f"Population CSV missing SA2 code col. Available cols: {list(pop.columns)}")

    # 选“总人口”列：按常见命名优先级匹配
    pop_value_col = pick_col(pop.columns, [
        "Tot_P_P", "Total_Persons", "Total_Persons_People", "Persons", "Tot_P"
    ])
    if not pop_value_col:
        # 兜底：如果找不到，就把可能的人口列打印出来让你确认
        numeric_cols = [c for c in pop.columns if pd.api.types.is_numeric_dtype(pop[c])]
        raise RuntimeError(
            "Cannot find total population column in POP_CSV. "
            f"Try one of these numeric columns manually: {numeric_cols[:30]}"
        )

    pop = pop.rename(columns={pop_code_col: "SA2_CODE", pop_value_col: "population"})
    pop["SA2_CODE"] = pop["SA2_CODE"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(9)
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")

    pop = pop[["SA2_CODE", "population"]].dropna(subset=["population"]).drop_duplicates("SA2_CODE")

    print("▶ Merging...")
    merged = sa2.merge(pop, on="SA2_CODE", how="left")

    missing = merged["population"].isna().sum()
    print(f"▶ Done merge. Missing population rows = {missing} / {len(merged)}")
    if missing > 0:
        print("⚠️ If this is unexpectedly high, likely causes:")
        print("   1) POP_CSV only covers NSW but you didn't filter SA2 to NSW; OR")
        print("   2) SA2_CODE formats differ (should be 9-digit strings).")

    merged["population_density"] = merged["population"] / merged["area_sqkm"]

    print("▶ Saving...")
    merged[["SA2_CODE", "SA2_NAME", "population", "area_sqkm", "population_density"]].to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    merged.to_crs(epsg=4326).to_file(OUT_GEOJSON, driver="GeoJSON")

    print("✅ OK")
    print("CSV:", OUT_CSV)
    print("GeoJSON:", OUT_GEOJSON)

if __name__ == "__main__":
    main()
