"""Unit conversion utilities for wall detection system"""


class UnitConverter:
    """Convert between different measurement units"""
    
    # Conversion constants
    METERS_TO_FEET = 3.28084
    METERS_TO_INCHES = 39.3701
    SQM_TO_SQFT = 10.764
    SQM_TO_SQIN = 1550.0031
    
    @staticmethod
    def meters_to_feet(meters):
        """Convert meters to feet"""
        return meters * UnitConverter.METERS_TO_FEET
    
    @staticmethod
    def meters_to_inches(meters):
        """Convert meters to inches"""
        return meters * UnitConverter.METERS_TO_INCHES
    
    @staticmethod
    def sqm_to_sqft(sqm):
        """Convert square meters to square feet"""
        return sqm * UnitConverter.SQM_TO_SQFT
    
    @staticmethod
    def sqm_to_sqin(sqm):
        """Convert square meters to square inches"""
        return sqm * UnitConverter.SQM_TO_SQIN
    
    @staticmethod
    def format_area(area_m2):
        """
        Format area with both metric and imperial units
        
        Args:
            area_m2 (float): Area in square meters
            
        Returns:
            str: Formatted string with both units
        """
        area_sqft = UnitConverter.sqm_to_sqft(area_m2)
        return f"{area_m2:.2f} m² ({area_sqft:.2f} sq ft)"
    
    @staticmethod
    def format_distance(distance_m):
        """
        Format distance with both metric and imperial units
        
        Args:
            distance_m (float): Distance in meters
            
        Returns:
            str: Formatted string with both units
        """
        distance_ft = UnitConverter.meters_to_feet(distance_m)
        return f"{distance_m:.2f} m ({distance_ft:.2f} ft)"


# Quick conversion functions
def m2_to_sqft(sqm):
    """Shorthand: Convert square meters to square feet"""
    return UnitConverter.sqm_to_sqft(sqm)


def m_to_ft(meters):
    """Shorthand: Convert meters to feet"""
    return UnitConverter.meters_to_feet(meters)


def format_area(area_m2):
    """Shorthand: Format area in both units"""
    return UnitConverter.format_area(area_m2)


def format_distance(distance_m):
    """Shorthand: Format distance in both units"""
    return UnitConverter.format_distance(distance_m)
