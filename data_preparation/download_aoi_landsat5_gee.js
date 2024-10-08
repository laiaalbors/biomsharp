/*
LANDSAT-5

This code is intended to be run in the Google Earth Engine (GEE) Code Editor.
To execute this code, copy and paste it into the GEE Code Editor at https://code.earthengine.google.com/
Ensure you have an Earth Engine account and the appropriate permissions to access Landsat-5 data.
*/

var L5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2");
var RegionEurope = ee.Geometry.Rectangle([12.0, 46.000000000040004, 15.99999999996, 50.0]);

var L5_Filtered = L5.filterBounds(RegionEurope)
                       .filterMetadata("CLOUD_COVER", "less_than", 15)
                       .filterDate('2010-01-01', '2010-12-31');

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true);
}

var L5_Filtered_scaled = L5_Filtered.map(applyScaleFactors);

// Apply the median to combine the different images into a single one.
var Median_Image = L5_Filtered_scaled.median();

// Select the specific bands you want to export
var selectedBands = Median_Image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']);

// Export the selected bands to Google Drive
Export.image.toDrive({
  image: selectedBands,
  description: "Europe_Small_Landsat5_2010",
  region: RegionEurope,
  scale: 30,
  crs: "EPSG:4326",
  maxPixels: 221294352
});