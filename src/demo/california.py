from __future__ import annotations


CALIFORNIA_AGGREGATE_COLUMNS = {
    "total_rooms", "total_bedrooms", "population", "households", "median_house_value"
}
CALIFORNIA_ROLE_OVERRIDES = {
    "total_rooms": ("rooms", "Rooms", "Aggregate Rooms"),
    "total_bedrooms": ("bedrooms", "Rooms", "Aggregate Bedrooms"),
    "housing_median_age": ("building_age", "Building", "Median Housing Age"),
    "population": ("population", "Neighborhood", "Population"),
    "households": ("households", "Neighborhood", "Households"),
    "median_income": ("median_income", "Neighborhood", "Median Income"),
    "median_house_value": ("sale_price", "Financial", "Median House Value"),
    "ocean_proximity": ("location_context", "Location", "Ocean Proximity"),
}


def is_california_aggregate(columns: set[str]) -> bool:
    return CALIFORNIA_AGGREGATE_COLUMNS.issubset(columns)


def role_override(normalized_column: str) -> tuple[str, str, str] | None:
    return CALIFORNIA_ROLE_OVERRIDES.get(normalized_column)
