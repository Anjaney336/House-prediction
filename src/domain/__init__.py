"""Domain intelligence for real-estate and generic regression datasets."""

from src.domain.domain_detector import DomainAnalysis, analyze_domain
from src.domain.target_detector import TargetCandidate, detect_targets

__all__ = ["DomainAnalysis", "TargetCandidate", "analyze_domain", "detect_targets"]
