# PixelHit BDT Transformer

This folder now contains a k4FWCore MultiTransformer that evaluates a TMVA BDT XML
weights on EDM4hep `TrackerHitPlane` objects.

Files added:
- `include/PixelHitBDTAlg.h`
- `src/PixelHitBDTAlg.cpp`
- `CMakeLists.txt`
- `options/run_pixel_hit_bdt.py`

## What It Produces Per Event

For each input event:
- `OutputFilteredHitCollection`: subset of hits that pass an optional BDT cut.

Inputs expected by the algorithm:
- `InputHitCollectionName` (default: `VXDBarrelHits`)
- `InputRawHitRelationCollectionName` (default: `VTXRawHitRelations`)
- `InputSimHitCollectionName` (default: `VertexBarrel`)

## Important EDM4hep Note

The training tuple used LCIO cluster-constituent information (pixel-level energies/times,
skewness, full cluster size from raw pixels).

This implementation now uses the digitizer raw-hit relation collection
(`TrackerHitSimTrackerHitLinkCollection`) together with the corresponding
`SimTrackerHitCollection` to recover cluster constituents and compute
the BDT variables from those real raw hits.

## Build

This is a standalone CMake snippet that can be built directly in a key4hep environment or copied
into an existing package.

Example:

```bash
mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . -t install -j $(nproc)
```

Main properties:
- `WeightsFile`
- `MethodName` (default: `BDT`)
- `InputHitCollectionName`
- `InputRawHitRelationCollectionName`
- `InputSimHitCollectionName`
- `OutputFilteredHitCollectionName`
- `MinBdtScore`
- `NumThreads`
