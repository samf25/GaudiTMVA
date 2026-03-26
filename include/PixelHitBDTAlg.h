#pragma once

#include <Gaudi/Property.h>

#include <TMVA/Reader.h>

#include <edm4hep/SimTrackerHit.h>
#include <edm4hep/SimTrackerHitCollection.h>
#include <edm4hep/TrackerHitPlaneCollection.h>
#include <edm4hep/TrackerHitSimTrackerHitLinkCollection.h>

#include <k4FWCore/Transformer.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

/**
 * @brief Evaluate a TMVA BDT score for each TrackerHitPlane and filter by score.
 *
 * The algorithm takes both reconstructed TrackerHitPlane objects and the
 * raw-hit relation collection from the digitiser to reconstruct all 33 TMVA
 * input features from real cluster constituents.
 */
struct PixelHitBDTAlg final
    : k4FWCore::MultiTransformer<std::tuple<edm4hep::TrackerHitPlaneCollection>(
          const edm4hep::TrackerHitPlaneCollection&,
          const edm4hep::TrackerHitSimTrackerHitLinkCollection&,
          const edm4hep::SimTrackerHitCollection&)> {
public:
  PixelHitBDTAlg(const std::string& name, ISvcLocator* svcLoc);

  StatusCode initialize() final;
  std::tuple<edm4hep::TrackerHitPlaneCollection> operator()(
      const edm4hep::TrackerHitPlaneCollection& hits,
      const edm4hep::TrackerHitSimTrackerHitLinkCollection& rawHitRelations,
      const edm4hep::SimTrackerHitCollection& simHits) const final;

private:
  static constexpr std::size_t kNVars = 33;
  static constexpr std::size_t kNPixelVars = 9;
  static const std::array<const char*, kNVars> s_varNames;

  void bindReaderVariables(TMVA::Reader& reader, std::array<float, kNVars>& vars) const;
  std::array<float, kNVars> buildFeatures(
      const edm4hep::TrackerHitPlane& hit,
      const std::vector<edm4hep::SimTrackerHit>& rawHits) const;

  static std::pair<float, float> getClusterSize(const std::vector<edm4hep::SimTrackerHit>& rawHits);
  static float getRMS(float clusterPos, const std::vector<edm4hep::SimTrackerHit>& rawHits, bool useX);
  static float getCov(float clusterPosX, float clusterPosY, const std::vector<edm4hep::SimTrackerHit>& rawHits);
  static float getSkew(float clusterPos, const std::vector<edm4hep::SimTrackerHit>& rawHits, bool useX);

  Gaudi::Property<std::string> m_weightsFile{
    this, "WeightsFile", "TMVAClassification_BDT.weights.xml", "Path to TMVA XML weights file"
  };
  Gaudi::Property<std::string> m_methodName{
    this, "MethodName", "BDT", "TMVA method name inside weights file"
  };
  Gaudi::Property<float> m_minBdtScore{
    this, "MinBdtScore", 0.0f, "Minimum BDT score required to keep a hit"
  };
  Gaudi::Property<int> m_numThreads{
    this, "NumThreads", 1, "Number of threads for per-hit BDT evaluation"
  };
};
