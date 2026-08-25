# Data Dictionary

This project supports both the bundled California Housing benchmark and property-listing style datasets. The active app detects available columns and trains only on fields that exist in the uploaded dataset.

## California Housing Benchmark

| Column | Type | Role | Description |
| --- | --- | --- | --- |
| `MedInc` | numeric | Feature | Median income in the census block group. Usually one of the strongest predictors of median house value. |
| `HouseAge` | numeric | Feature | Median age of houses in the census block group. |
| `AveRooms` | numeric | Feature | Average number of rooms per household. |
| `AveBedrms` | numeric | Feature | Average number of bedrooms per household. |
| `Population` | numeric | Feature | Population of the census block group. |
| `AveOccup` | numeric | Feature | Average household occupancy. |
| `Latitude` | numeric | Feature | Latitude of the census block group. |
| `Longitude` | numeric | Feature | Longitude of the census block group. |
| `MedHouseVal` | numeric | Target | Median house value for the census block group. This is an aggregate target, not an individual property sale price. |

## Property Listing Dataset Pattern

| Column pattern | Type | Role | Description |
| --- | --- | --- | --- |
| `price`, `sale_price`, `listing_price`, `transaction_value`, `price_inr` | numeric | Target candidate | Price or value field selected only after target confirmation. |
| `area`, `area_sqft`, `built_up_area`, `plot_area` | numeric | Feature | Size measurement used as a core structural predictor. |
| `bedrooms`, `bathrooms`, `rooms` | numeric | Feature | Residential structure and layout fields. |
| `property_type`, `asset_type` | categorical | Feature | House, apartment, villa, land, commercial, rental, or related asset category. |
| `city`, `locality`, `region`, `postal_code` | categorical/location | Feature | Location fields used for market segmentation and model routing. |
| `latitude`, `longitude` | numeric/location | Feature | Geographic coordinates when available and valid. |
| `year_built`, `listing_date`, `sale_date` | numeric/datetime | Feature | Temporal or age-related fields. Date fields are handled through datetime-aware preprocessing. |
| `listing_id`, `property_id`, `url`, `seller_id` | identifier | Excluded by default | Identifier-like fields are blocked unless intentionally retained. |

## Modeling Notes

- Raw columns are not renamed or overwritten.
- The target is never included as a training feature.
- Missing feature values are handled by fitted preprocessing inside the model pipeline.
- Rows with missing target values are removed before model training.
- Price-derived leakage fields and near-perfect target proxies are flagged before training.
- The California dataset is treated as a geographic aggregate benchmark, not an exact individual-property valuation dataset.
