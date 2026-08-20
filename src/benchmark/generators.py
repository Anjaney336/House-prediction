from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketConfig:
    base_market_level: float = 1.0
    location_premium: float = 1.0
    area_coefficient: float = 1.0
    bedroom_coefficient: float = 1.0
    age_depreciation: float = 1.0
    amenity_premium: float = 1.0
    parking_premium: float = 1.0
    luxury_premium: float = 1.0
    market_inflation: float = 0.055
    noise: float = 0.08


@dataclass(frozen=True)
class SyntheticDataset:
    frame: pd.DataFrame
    dataset_type: str
    target: str
    seed: int
    parameters: dict
    ground_truth: dict[str, float]

    def manifest(self) -> dict:
        return {
            "dataset_type": self.dataset_type,
            "target": self.target,
            "seed": self.seed,
            "rows": len(self.frame),
            "columns": len(self.frame.columns),
            "parameters": self.parameters,
            "ground_truth": self.ground_truth,
        }


CITY_PROFILES = {
    "Metro A": (28.61, 77.21, 1.35),
    "Metro B": (19.08, 72.88, 1.28),
    "City C": (12.97, 77.59, 1.18),
    "City D": (22.57, 88.36, 0.92),
}
LOCALITY_PREMIUM = {"Prime": 1.35, "Central": 1.18, "Established": 1.03, "Emerging": 0.82}


def _context(rng: np.random.Generator, rows: int, config: MarketConfig):
    cities = rng.choice(list(CITY_PROFILES), rows, p=[0.3, 0.27, 0.25, 0.18])
    localities = rng.choice(list(LOCALITY_PREMIUM), rows, p=[0.16, 0.27, 0.36, 0.21])
    city_factor = np.array([CITY_PROFILES[value][2] for value in cities])
    locality_factor = np.array([LOCALITY_PREMIUM[value] for value in localities]) ** config.location_premium
    latitude = np.array([CITY_PROFILES[value][0] for value in cities]) + rng.normal(0, 0.065, rows)
    longitude = np.array([CITY_PROFILES[value][1] for value in cities]) + rng.normal(0, 0.065, rows)
    dates = pd.to_datetime(rng.integers(pd.Timestamp("2020-01-01").value // 10**9, pd.Timestamp("2026-12-31").value // 10**9, rows), unit="s")
    inflation = (1 + config.market_inflation) ** (dates.year.to_numpy() - 2020)
    return cities, localities, city_factor, locality_factor, latitude, longitude, dates, inflation


def _market_noise(rng: np.random.Generator, signal: np.ndarray, noise: float, hetero: np.ndarray | float = 1.0):
    return signal * np.exp(rng.normal(-0.5 * noise**2, noise * hetero, len(signal)))


def generate_apartments(rows: int = 500, seed: int = 42, config: MarketConfig | None = None) -> SyntheticDataset:
    config, rng = config or MarketConfig(), np.random.default_rng(seed)
    cities, locality, city_f, loc_f, lat, lon, dates, inflation = _context(rng, rows, config)
    area = np.clip(rng.lognormal(np.log(1150), 0.34, rows), 320, 4200)
    bedrooms = np.clip(np.rint(area / 480 + rng.normal(0.4, 0.55, rows)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms + rng.choice([-1, 0, 0, 1], rows), 1, 7)
    total_floors = rng.integers(4, 35, rows)
    floor = np.array([rng.integers(0, maximum + 1) for maximum in total_floors])
    age = np.clip(rng.gamma(2.2, 6.0, rows), 0, 55)
    parking = rng.binomial(2, np.clip(0.35 + area / 5000, 0.35, 0.9))
    furnishing = rng.choice(["Unfurnished", "Semi-Furnished", "Furnished"], rows, p=[0.28, 0.48, 0.24])
    amenities = np.clip(rng.poisson(4.5 + (locality == "Prime") * 2.0, rows), 0, 14)
    center = np.clip(rng.gamma(2.0, 4.0, rows) / loc_f, 0.3, 35)
    metro = np.clip(rng.gamma(1.6, 1.4, rows), 0.05, 12)
    developer = rng.choice(["Local", "Trusted", "Premium"], rows, p=[0.48, 0.37, 0.15])
    floor_effect = 1 + 0.07 * np.sin(np.pi * floor / np.maximum(total_floors, 1)) - 0.02 * (floor == 0)
    amenity_effect = 1 + config.amenity_premium * (0.018 * amenities + 0.0008 * amenities**2)
    area_effect = (area ** 0.92) * (1 + 0.035 * bedrooms * config.bedroom_coefficient)
    signal = (
        9100 * config.base_market_level * city_f * loc_f * area_effect * floor_effect * amenity_effect
        * (1 + 0.035 * bathrooms + 0.045 * parking * config.parking_premium)
        * np.where(developer == "Premium", 1.13, np.where(developer == "Trusted", 1.06, 1.0))
        * np.exp(-0.009 * age * config.age_depreciation - 0.012 * center - 0.018 * metro)
        * inflation
    )
    price = _market_noise(rng, signal, config.noise, 0.7 + area / area.mean() * 0.3)
    frame = pd.DataFrame({
        "property_id": [f"APT-{seed}-{i:06d}" for i in range(rows)], "property_type": "Apartment",
        "city": cities, "locality": locality, "latitude": lat, "longitude": lon,
        "area_sqft": area.round(1), "bedrooms": bedrooms, "bathrooms": bathrooms,
        "floor": floor, "total_floors": total_floors, "building_age": age.round(1),
        "parking": parking, "furnishing": furnishing, "amenities": amenities,
        "distance_to_center": center.round(2), "distance_to_metro": metro.round(2),
        "developer": developer, "transaction_date": dates, "sale_price": price.round(0),
    })
    truth = {"area_sqft": 1.0, "locality": 0.95, "city": 0.85, "building_age": 0.55, "amenities": 0.50, "distance_to_center": 0.45, "parking": 0.3}
    return SyntheticDataset(frame, "apartments", "sale_price", seed, asdict(config), truth)


def generate_villas(rows: int = 500, seed: int = 42, config: MarketConfig | None = None) -> SyntheticDataset:
    config, rng = config or MarketConfig(), np.random.default_rng(seed)
    cities, locality, city_f, loc_f, lat, lon, dates, inflation = _context(rng, rows, config)
    plot = np.clip(rng.lognormal(np.log(2600), 0.48, rows), 700, 15000)
    built = np.clip(plot * rng.uniform(0.32, 0.78, rows), 600, 9000)
    beds = np.clip(np.rint(built / 720 + rng.normal(1, 0.6, rows)), 2, 10).astype(int)
    baths = np.clip(beds + rng.choice([-1, 0, 1, 2], rows), 2, 12)
    floors = rng.integers(1, 5, rows); garden = rng.binomial(1, 0.72, rows); pool = rng.binomial(1, 0.28, rows)
    parking = rng.integers(1, 6, rows); security = rng.binomial(1, 0.68, rows)
    age = np.clip(rng.gamma(1.8, 7, rows), 0, 60); road = rng.choice([20, 30, 40, 60, 80], rows)
    corner = rng.binomial(1, 0.22, rows); luxury = rng.choice(["Standard", "Premium", "Ultra"], rows, p=[0.48, 0.38, 0.14])
    luxury_f = np.where(luxury == "Ultra", 1.38, np.where(luxury == "Premium", 1.16, 1.0)) ** config.luxury_premium
    pool_effect = 1 + pool * (0.035 + 0.11 * (luxury == "Ultra"))
    signal = (
        11500 * config.base_market_level * city_f * loc_f * (plot ** 0.58) * (built ** 0.48)
        * (1 + 0.025 * beds + 0.018 * baths + 0.035 * parking * config.parking_premium)
        * (1 + 0.07 * garden + 0.045 * security + 0.0014 * road + 0.065 * corner)
        * pool_effect * luxury_f * np.exp(-0.007 * age * config.age_depreciation) * inflation
    )
    price = _market_noise(rng, signal, config.noise * 1.15, 0.8 + luxury_f * 0.2)
    frame = pd.DataFrame({
        "property_id": [f"VIL-{seed}-{i:06d}" for i in range(rows)], "property_type": "Villa",
        "city": cities, "location": locality, "latitude": lat, "longitude": lon,
        "plot_area": plot.round(1), "built_up_area": built.round(1), "bedrooms": beds,
        "bathrooms": baths, "floors": floors, "garden": garden, "pool": pool,
        "parking": parking, "security": security, "construction_age": age.round(1),
        "road_width": road, "corner_plot": corner, "luxury_level": luxury,
        "transaction_date": dates, "sale_price": price.round(0),
    })
    truth = {"plot_area": 1.0, "built_up_area": 0.95, "location": 0.9, "city": 0.8, "luxury_level": 0.65, "pool": 0.4, "construction_age": 0.4}
    return SyntheticDataset(frame, "villas", "sale_price", seed, asdict(config), truth)


def generate_mixed_residential(rows: int = 500, seed: int = 42, config: MarketConfig | None = None) -> SyntheticDataset:
    config, rng = config or MarketConfig(), np.random.default_rng(seed)
    cities, locality, city_f, loc_f, lat, lon, dates, inflation = _context(rng, rows, config)
    kinds = rng.choice(["Apartment", "Villa", "Townhouse", "Independent House", "Studio"], rows, p=[0.38, 0.15, 0.18, 0.2, 0.09])
    size_base = np.select([kinds == "Studio", kinds == "Apartment", kinds == "Townhouse", kinds == "Villa"], [520, 1150, 1800, 3200], default=2200)
    area = np.clip(rng.lognormal(np.log(size_base), 0.3), 250, 9000)
    plot = np.where(np.isin(kinds, ["Villa", "Townhouse", "Independent House"]), area * rng.uniform(1.05, 2.2, rows), 0)
    bedrooms = np.clip(np.rint(area / np.where(kinds == "Studio", 900, 550) + rng.normal(0.2, 0.5, rows)), 1, 9).astype(int)
    bathrooms = np.clip(bedrooms + rng.choice([-1, 0, 0, 1], rows), 1, 10)
    floor = rng.integers(0, 25, rows); lift = ((kinds == "Apartment") & (floor > 3)) | (rng.random(rows) < 0.25)
    garden = np.isin(kinds, ["Villa", "Townhouse", "Independent House"]) & (rng.random(rows) < 0.58)
    parking = rng.integers(0, 4, rows); road = rng.choice([15, 20, 30, 40, 60], rows); amenities = rng.integers(0, 12, rows)
    age = np.clip(rng.gamma(2, 6, rows), 0, 60)
    type_f = np.select([kinds == "Studio", kinds == "Apartment", kinds == "Townhouse", kinds == "Villa"], [1.08, 1.0, 1.08, 1.2], default=1.03)
    interaction = 1 + (kinds == "Apartment") * (0.006 * floor + 0.018 * amenities + 0.05 * lift) + (kinds == "Villa") * (0.000025 * plot + 0.055 * garden + 0.018 * road) + (kinds == "Studio") * (0.16 * (locality == "Prime"))
    signal = 9800 * config.base_market_level * city_f * loc_f * area**0.94 * type_f * interaction * (1 + 0.03 * bedrooms + 0.025 * bathrooms + 0.04 * parking) * np.exp(-0.008 * age) * inflation
    price = _market_noise(rng, signal, config.noise * 1.2, 1.0)
    frame = pd.DataFrame({
        "property_id": [f"MIX-{seed}-{i:06d}" for i in range(rows)], "property_type": kinds,
        "city": cities, "locality": locality, "latitude": lat, "longitude": lon,
        "area_sqft": area.round(1), "plot_area": plot.round(1), "bedrooms": bedrooms,
        "bathrooms": bathrooms, "floor": floor, "lift": lift.astype(int), "garden": garden.astype(int),
        "parking": parking, "road_width": road, "amenities": amenities, "building_age": age.round(1),
        "transaction_date": dates, "sale_price": price.round(0),
    })
    truth = {"area_sqft": 1.0, "property_type": 0.95, "locality": 0.9, "city": 0.8, "plot_area": 0.6, "amenities": 0.4}
    return SyntheticDataset(frame, "mixed_residential", "sale_price", seed, asdict(config), truth)


def generate_land(rows: int = 500, seed: int = 42, config: MarketConfig | None = None) -> SyntheticDataset:
    config, rng = config or MarketConfig(), np.random.default_rng(seed)
    cities, locality, city_f, loc_f, lat, lon, dates, inflation = _context(rng, rows, config)
    area = np.clip(rng.lognormal(np.log(3600), 0.7, rows), 400, 40000); road = rng.choice([12, 20, 30, 40, 60, 80], rows)
    corner = rng.binomial(1, 0.2, rows); frontage = np.sqrt(area) * rng.uniform(0.6, 1.8, rows)
    zoning = rng.choice(["Residential", "Mixed Use", "Commercial", "Agricultural"], rows, p=[0.48, 0.22, 0.16, 0.14])
    commercial = np.clip(rng.beta(2, 3, rows) + (zoning == "Commercial") * 0.45, 0, 1)
    status = rng.choice(["Raw", "Approved", "Serviced"], rows, p=[0.3, 0.42, 0.28])
    highway = rng.gamma(2, 4, rows); city_distance = rng.gamma(2.2, 6, rows)
    zone_f = np.select([zoning == "Commercial", zoning == "Mixed Use", zoning == "Agricultural"], [1.55, 1.28, 0.58], default=1.0)
    status_f = np.select([status == "Serviced", status == "Approved"], [1.22, 1.1], default=0.86)
    signal = 4200 * config.base_market_level * city_f * loc_f * area**0.97 * zone_f * status_f * (1 + 0.0018 * road + 0.07 * corner + 0.002 * frontage + 0.2 * commercial) * np.exp(-0.012 * highway - 0.018 * city_distance) * inflation
    price = _market_noise(rng, signal, config.noise * 1.25)
    frame = pd.DataFrame({"property_id": [f"LND-{seed}-{i:06d}" for i in range(rows)], "property_type": "Land Plot", "city": cities, "location": locality, "latitude": lat, "longitude": lon, "plot_area": area.round(1), "road_width": road, "corner_plot": corner, "frontage": frontage.round(1), "zoning": zoning, "commercial_potential": commercial.round(3), "development_status": status, "distance_to_highway": highway.round(2), "distance_to_city": city_distance.round(2), "transaction_date": dates, "sale_price": price.round(0)})
    truth = {"plot_area": 1.0, "zoning": 0.95, "location": 0.85, "development_status": 0.7, "distance_to_city": 0.55, "road_width": 0.4}
    return SyntheticDataset(frame, "land", "sale_price", seed, asdict(config), truth)


def generate_commercial(rows: int = 500, seed: int = 42, config: MarketConfig | None = None) -> SyntheticDataset:
    config, rng = config or MarketConfig(), np.random.default_rng(seed)
    cities, locality, city_f, loc_f, lat, lon, dates, inflation = _context(rng, rows, config)
    kinds = rng.choice(["Office", "Retail", "Showroom", "Warehouse"], rows, p=[0.38, 0.28, 0.12, 0.22])
    area = np.clip(rng.lognormal(np.log(np.where(kinds == "Warehouse", 5000, 1700)), 0.55), 350, 30000)
    floor = np.where(kinds == "Warehouse", 0, rng.integers(0, 20, rows)); parking = rng.integers(0, 15, rows)
    footfall = np.clip(rng.lognormal(5.5, 0.7, rows) * np.where(np.isin(kinds, ["Retail", "Showroom"]), 1.7, 0.65), 20, 5000)
    visibility = rng.integers(1, 11, rows); road = rng.choice([20, 30, 40, 60, 80, 120], rows); age = np.clip(rng.gamma(2, 7, rows), 0, 65)
    lease = rng.choice(["Vacant", "Owner Occupied", "Leased"], rows, p=[0.16, 0.34, 0.5])
    type_f = np.select([kinds == "Retail", kinds == "Showroom", kinds == "Warehouse"], [1.18, 1.3, 0.72], default=1.0)
    lease_f = np.select([lease == "Leased", lease == "Vacant"], [1.1, 0.88], default=1.0)
    signal = 12500 * config.base_market_level * city_f * loc_f * area**0.93 * type_f * lease_f * (1 + 0.02 * parking + 0.00012 * footfall * np.isin(kinds, ["Retail", "Showroom"]) + 0.025 * visibility + 0.0015 * road) * np.exp(-0.006 * age) * inflation
    price = _market_noise(rng, signal, config.noise * 1.3)
    frame = pd.DataFrame({"property_id": [f"COM-{seed}-{i:06d}" for i in range(rows)], "property_type": kinds, "city": cities, "location": locality, "latitude": lat, "longitude": lon, "usable_area": area.round(1), "floor": floor, "parking": parking, "footfall": footfall.round(0), "visibility": visibility, "road_width": road, "building_age": age.round(1), "lease_status": lease, "transaction_date": dates, "sale_price": price.round(0)})
    truth = {"usable_area": 1.0, "property_type": 0.9, "location": 0.85, "footfall": 0.65, "visibility": 0.5, "lease_status": 0.45}
    return SyntheticDataset(frame, "commercial", "sale_price", seed, asdict(config), truth)


def generate_rentals(rows: int = 500, seed: int = 42, config: MarketConfig | None = None) -> SyntheticDataset:
    config, rng = config or MarketConfig(), np.random.default_rng(seed)
    cities, locality, city_f, loc_f, lat, lon, dates, inflation = _context(rng, rows, config)
    kinds = rng.choice(["Apartment", "House", "Studio"], rows, p=[0.62, 0.22, 0.16])
    area = np.clip(rng.lognormal(np.log(np.where(kinds == "Studio", 500, np.where(kinds == "House", 1900, 1050))), 0.32), 250, 5000)
    beds = np.clip(np.rint(area / 550), 1, 7).astype(int); baths = np.clip(beds + rng.choice([-1, 0, 0, 1], rows), 1, 8)
    furnishing = rng.choice(["Unfurnished", "Semi-Furnished", "Furnished"], rows, p=[0.24, 0.5, 0.26])
    parking = rng.integers(0, 3, rows); amenities = rng.integers(0, 12, rows); age = np.clip(rng.gamma(2, 6, rows), 0, 55)
    furnish_f = np.select([furnishing == "Furnished", furnishing == "Semi-Furnished"], [1.16, 1.07], default=1.0)
    signal = 31 * config.base_market_level * city_f * loc_f * area**0.91 * furnish_f * (1 + 0.025 * beds + 0.022 * baths + 0.045 * parking + 0.016 * amenities) * np.exp(-0.006 * age) * inflation
    rent = _market_noise(rng, signal, config.noise)
    frame = pd.DataFrame({"property_id": [f"RNT-{seed}-{i:06d}" for i in range(rows)], "property_type": kinds, "city": cities, "location": locality, "latitude": lat, "longitude": lon, "area": area.round(1), "bedrooms": beds, "bathrooms": baths, "furnishing": furnishing, "parking": parking, "amenities": amenities, "building_age": age.round(1), "transaction_date": dates, "monthly_rent": rent.round(0)})
    truth = {"area": 1.0, "location": 0.95, "city": 0.85, "furnishing": 0.55, "amenities": 0.4, "building_age": 0.35}
    return SyntheticDataset(frame, "rentals", "monthly_rent", seed, asdict(config), truth)


GENERATORS: dict[str, Callable[..., SyntheticDataset]] = {
    "Apartments": generate_apartments, "Villas": generate_villas,
    "Mixed Residential": generate_mixed_residential, "Land / Plots": generate_land,
    "Commercial": generate_commercial, "Rentals": generate_rentals,
}


def generate_dataset(kind: str, rows: int = 500, seed: int = 42, config: MarketConfig | None = None) -> SyntheticDataset:
    if kind not in GENERATORS:
        raise ValueError(f"Unknown synthetic dataset '{kind}'. Available: {', '.join(GENERATORS)}")
    if rows < 50:
        raise ValueError("Synthetic benchmarks require at least 50 rows.")
    return GENERATORS[kind](rows=rows, seed=seed, config=config)


def save_synthetic_dataset(dataset: SyntheticDataset, directory: str | Path) -> tuple[Path, Path]:
    import json

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"synthetic_{dataset.dataset_type}_seed{dataset.seed}"
    csv_path, manifest_path = output / f"{stem}.csv", output / f"{stem}.json"
    dataset.frame.to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps(dataset.manifest(), indent=2, default=str), encoding="utf-8")
    return csv_path, manifest_path
