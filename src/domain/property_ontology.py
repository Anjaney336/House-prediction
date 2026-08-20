from __future__ import annotations

import re
from dataclasses import dataclass


def normalize_name(name: str) -> str:
    """Normalize a raw column name for conservative semantic matching."""
    value = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return re.sub(r"_+", "_", value)


@dataclass(frozen=True)
class OntologyEntry:
    role: str
    group: str
    label: str
    aliases: tuple[str, ...]


ONTOLOGY = (
    OntologyEntry("property_id", "Property", "Property ID", ("property_id", "listing_id", "listing_ref", "parcel_id")),
    OntologyEntry("property_type", "Property", "Property Type", ("property_type", "property_category", "asset_type", "type_of_property")),
    OntologyEntry("sub_property_type", "Property", "Property Subtype", ("sub_property_type", "property_subtype", "subtype")),
    OntologyEntry("country", "Location", "Country", ("country", "country_name")),
    OntologyEntry("state", "Location", "State", ("state", "province", "region")),
    OntologyEntry("city", "Location", "City", ("city", "town", "municipality")),
    OntologyEntry("locality", "Location", "Locality", ("locality", "locality_name", "neighborhood", "neighbourhood", "location", "suburb")),
    OntologyEntry("sector", "Location", "Sector", ("sector", "sector_name")),
    OntologyEntry("postal_code", "Location", "Postal Code", ("postal_code", "postcode", "zip", "zipcode", "pin_code", "pincode")),
    OntologyEntry("latitude", "Location", "Latitude", ("latitude", "lat")),
    OntologyEntry("longitude", "Location", "Longitude", ("longitude", "lon", "lng")),
    OntologyEntry("area", "Size", "Area", ("area", "area_sqft", "sqft", "square_feet", "square_foot", "size", "total_area", "super_area", "super_area_sqft", "office_area", "retail_area", "warehouse_area")),
    OntologyEntry("built_up_area", "Size", "Built-up Area", ("built_up_area", "builtup_area", "built_up_sqft", "covered_area")),
    OntologyEntry("carpet_area", "Size", "Carpet Area", ("carpet_area", "carpet_sqft")),
    OntologyEntry("plot_area", "Size", "Plot Area", ("plot_area", "land_area", "lot_size", "lot_area")),
    OntologyEntry("bedrooms", "Rooms", "Bedrooms", ("bedrooms", "bedroom", "beds", "bed", "bhk", "num_bedrooms")),
    OntologyEntry("bathrooms", "Rooms", "Bathrooms", ("bathrooms", "bathroom", "baths", "bath", "num_bathrooms")),
    OntologyEntry("balconies", "Rooms", "Balconies", ("balconies", "balcony", "num_balconies")),
    OntologyEntry("rooms", "Rooms", "Rooms", ("rooms", "room_count")),
    OntologyEntry("floor", "Building", "Floor", ("floor", "floor_number", "floor_no", "storey")),
    OntologyEntry("total_floors", "Building", "Total Floors", ("total_floors", "floors", "building_floors")),
    OntologyEntry("building_age", "Building", "Building Age", ("building_age", "property_age", "age", "age_years")),
    OntologyEntry("construction_year", "Building", "Construction Year", ("construction_year", "year_built", "built_year")),
    OntologyEntry("furnishing", "Property", "Furnishing", ("furnishing", "furnished", "furnishing_status")),
    OntologyEntry("condition", "Property", "Property Condition", ("condition", "property_condition", "renovation_status")),
    OntologyEntry("parking", "Amenities", "Parking", ("parking", "parking_spaces", "parking_slots", "garage", "car_parking")),
    OntologyEntry("amenities", "Amenities", "Amenities", ("amenities", "amenities_score", "facility", "facilities")),
    OntologyEntry("facing", "Property", "Facing", ("facing", "orientation")),
    OntologyEntry("metro_distance", "Accessibility", "Metro Distance", ("metro_distance", "metro_distance_km", "transit_distance")),
    OntologyEntry("highway_distance", "Accessibility", "Highway Distance", ("highway_distance", "highway_distance_km", "distance_to_highway")),
    OntologyEntry("school_distance", "Accessibility", "School Distance", ("school_distance", "school_distance_km")),
    OntologyEntry("hospital_distance", "Accessibility", "Hospital Distance", ("hospital_distance", "hospital_distance_km")),
    OntologyEntry("city_center_distance", "Accessibility", "City Center Distance", ("city_center_distance", "city_center_distance_km", "distance_to_city", "distance_to_center")),
    OntologyEntry("rera_registered", "Legal / Ownership", "RERA Registered", ("rera", "rera_registered", "is_rera_registered")),
    OntologyEntry("ownership", "Legal / Ownership", "Ownership", ("ownership", "ownership_type", "title_status")),
    OntologyEntry("developer", "Property", "Developer", ("developer", "builder")),
    OntologyEntry("project", "Property", "Project", ("project", "project_name", "society", "building")),
    OntologyEntry("road_width", "Land", "Road Width", ("road_width", "road_width_ft")),
    OntologyEntry("frontage", "Land", "Frontage", ("frontage", "plot_frontage")),
    OntologyEntry("corner_plot", "Land", "Corner Plot", ("corner_plot", "is_corner")),
    OntologyEntry("zoning", "Land", "Zoning", ("zoning", "zone", "land_use")),
    OntologyEntry("sale_price", "Financial", "Sale Price", ("price", "price_inr", "price_usd", "sale_price", "sale_price_usd", "selling_price", "listing_price", "asking_price", "transaction_price", "transaction_value", "property_value", "sold_value", "sale_amount")),
    OntologyEntry("rent", "Financial", "Rent", ("rent", "monthly_rent", "annual_rent", "lease_rent")),
    OntologyEntry("price_per_sqft", "Financial", "Price per sq ft", ("price_per_sqft", "price_sqft", "rate_per_sqft")),
    OntologyEntry("transaction_type", "Transaction", "Transaction Type", ("transaction_type", "sale_or_rent", "listing_type", "lease_status")),
    OntologyEntry("listing_date", "Transaction", "Listing Date", ("listing_date", "listed_at", "date_listed")),
    OntologyEntry("transaction_date", "Transaction", "Transaction Date", ("transaction_date", "sale_date", "sold_date", "closing_date")),
)

ALIAS_LOOKUP = {alias: entry for entry in ONTOLOGY for alias in entry.aliases}


def map_column(column: str) -> OntologyEntry | None:
    normalized = normalize_name(column)
    from src.demo.california import role_override
    demo_role = role_override(normalized)
    if demo_role:
        role, group, label = demo_role
        return OntologyEntry(role, group, label, (normalized,))
    if normalized in ALIAS_LOOKUP:
        return ALIAS_LOOKUP[normalized]
    tokens = set(normalized.split("_"))
    candidates = []
    for alias, entry in ALIAS_LOOKUP.items():
        alias_tokens = set(alias.split("_"))
        overlap = len(tokens & alias_tokens) / max(len(alias_tokens), 1)
        if overlap >= 0.8:
            candidates.append((overlap, len(alias), entry))
    return max(candidates, default=(0, 0, None))[2]


def schema_mapping(columns: list[str]) -> dict[str, dict[str, str]]:
    mapping = {}
    for column in columns:
        entry = map_column(column)
        mapping[column] = {
            "normalized_role": entry.role if entry else "other",
            "group": entry.group if entry else "Additional Model Inputs",
            "label": entry.label if entry else normalize_name(column).replace("_", " ").title(),
        }
    return mapping


VALUATION_SIGNAL_ROLES = {
    "property_type", "city", "locality", "sector", "postal_code", "latitude", "longitude", "area",
    "built_up_area", "carpet_area", "plot_area", "bedrooms", "bathrooms", "rooms", "floor",
    "building_age", "construction_year", "parking", "amenities", "metro_distance", "highway_distance",
    "school_distance", "hospital_distance", "city_center_distance", "road_width", "frontage", "zoning",
}
