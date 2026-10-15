# Dataset datasheet

- **Composition:** monthly site-level NASA POWER climate variables, NOAA indices, computed SPEI targets, and SPEI lag features.
- **Locations:** Guntur, Jaipur, Nagpur, Murray-Darling representative point, Addis Ababa, Phoenix, Seville, Pretoria.
- **Coverage:** 1981-01 through 2024-12; 8 × 528 = 4,224 site-month rows.
- **Sources:** NASA POWER monthly point service; NOAA CPC ONI and PSL DMI, NAO, PDO text series.
- **Target generation:** precipitation minus Hargreaves-Samani PET approximation; accumulated over 3/6/12 months and nonparametrically transformed by calendar month using only data through 2010-12.
- **Missingness:** targets remain masked in window construction. Raw source missing/fill values should be checked in dataset metadata before publication.
- **Known assumptions:** ONI seasonal values map to central months; monthly POWER values are point/grid-cell estimates; SPEI is computed rather than official SPEIbase.
- **Recommended release:** archive raw response hashes, retrieval date, provider versions, site coordinates, code commit, and derived CSV metadata with the paper. Observe provider terms; this repository does not redistribute raw data by default.
