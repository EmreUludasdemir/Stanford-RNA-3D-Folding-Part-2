# Stanford RNA 3D Folding 2: Hibrit TBM + Ab-Initio

Bu repo, Kaggle Stanford RNA 3D Folding yarismalari icin liderlik siralamasina oynayabilecek hizli bir temel cozum sunar.

## Hedef

RNA 3D yapilarini (C1' koordinatlari) hibrit bir yaklasimla tahmin etmek:
- Train etiketlerinden template tabanli modelleme (TBM)
- PDB tabanli template arama
- MSA ile uzak homolog zenginlestirme
- Coklu template ensemble
- Template yetersizliginde ab-initio A-form fallback

## Yarisma Linkleri

- Part 1: https://www.kaggle.com/competitions/stanford-rna-3d-folding
- Part 2: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
- Data: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data
- Code: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/code
- Models: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/models
- Leaderboard: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/leaderboard
- Submissions: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/submissions

## Son Durum Ozeti

| Script | Target | Submission Shape | Runtime |
| --- | --- | --- | --- |
| `stanford_rna_3d_folding_2.py` | `28` | `(9762, 18)` | `61.3 dakika` |

## Kisa Mimari

1. Train template + PDB `seqres` indekslerinin olusturulmasi
2. K-mer prefilter ile adaylarin daraltilmasi
3. Dizi hizalama (normal hedefte SW, uzun hedefte hizli yol)
4. Koordinat transferi (train label veya mmCIF)
5. 5 model uretimi (weighted ensemble + cesitlilik gurultusu)
6. Eksik koordinatlarin interpolation ile tamamlanmasi

## Performans Iyilestirmeleri

- Sparse k-mer benzerligi
- Uzun dizilerde `SequenceMatcher` hizli hizalama
- Uzun hedefler icin adaptif aday limiti
- PDB bazli mmCIF cache
- Deterministik tohum (`RANDOM_SEED=42`)

## Kaggle'da Calistirma

```bash
!PYDEVD_DISABLE_FILE_VALIDATION=1 python -Xfrozen_modules=off /kaggle/working/stanford_rna_3d_folding_2.py
```

Not:
- Debugger warning'leri genelde submission kalitesini etkilemez.
- Notebook sonunda gelen `nbconvert/traitlets` warning'leri cogunlukla export kaynaklidir.

## Yerelde Calistirma

```bash
pip install numpy pandas
python stanford_rna_3d_folding_2.py
```

Script varsayilan olarak Kaggle dizinlerini kullanir. Yerelde `INPUT_DIR` ve `OUTPUT_PATH` degerlerini guncellemelisin.

## Yol Haritasi

- [ ] Ultra uzun hedefler icin chunked alignment
- [ ] Stoichiometry-aware assembly refinement
- [ ] Daha guclu MSA agirliklandirma
- [ ] Hedef bazli profiler ciktilari
- [ ] Test kapsaminda artisim
- [ ] Deney takibi ve config preset sistemi

