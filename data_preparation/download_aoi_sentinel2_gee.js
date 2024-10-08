/*
SENTINEL-2

This code is intended to be run in the Google Earth Engine (GEE) Code Editor.
To execute this code, copy and paste it into the GEE Code Editor at https://code.earthengine.google.com/
Ensure you have an Earth Engine account and the appropriate permissions to access Sentinel-2 data.
*/

var S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var RegionEurope = ee.Geometry.Rectangle([12.0, 46.000000000040004, 15.99999999996, 50.0]);

var S2_Filtered = S2.filterBounds(RegionEurope)
                    .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 50)
                    .filterDate('2017-01-01', '2017-12-31');

// Apply a mask to filter out cloud and cloud shadow pixels.
function S2_maskClouds(image) {
  var scl = image.select('SCL');
  var cloudMask = scl.neq(8).and(scl.neq(9));
  return image.updateMask(cloudMask);
}

// Apply the cloud mask function to the image collection.
var cloudMaskedCollection = S2_Filtered.map(S2_maskClouds);

// Apply the median to combine the different images into a single one.
var Median_Image = cloudMaskedCollection.median();

// Export the resulting image.
Export.image.toDrive({
  image: Median_Image,
  description: "Europe_Small_Sentinel2",
  region: RegionEurope,
  scale: 25,
  crs: "EPSG:4326",
  maxPixels: 318638868
});