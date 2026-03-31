#pragma once

#include <Gaudi/Property.h>

#include <TMVA/Reader.h>

#include <edm4hep/SimTrackerHit.h>
#include <edm4hep/SimTrackerHitCollection.h>
#include <edm4hep/TrackerHitPlaneCollection.h>
#include <edm4hep/TrackerHitSimTrackerHitLinkCollection.h>

#include <k4FWCore/Transformer.h>

#include <podio/UserDataCollection.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

/**
 * @brief Evaluate TMVA BDT score for each TrackerHitPlane and export scores/labels for ROC studies.
 *
 * Inputs:
 * - TrackerHitPlaneCollection
 * - Raw-hit relation collection + raw SimTrackerHitCollection (for feature building)
 * - Truth relation collection + truth SimTrackerHitCollection (for label building)
 *
 * Outputs:
 * - TrackerHitPlane subset collection containing only hits above a configurable BDT cut
 * - Per-hit BDT score collection (float)
 * - Per-hit truth label collection (float, 0 or 1)
 */
struct PixelHitBDTTestAlg final
    : k4FWCore::MultiTransformer<std::tuple<edm4hep::TrackerHitPlaneCollection,
                                            podio::UserDataCollection<float>,
                                            podio::UserDataCollection<float>>(
          const edm4hep::TrackerHitPlaneCollection&,
          const edm4hep::TrackerHitSimTrackerHitLinkCollection&,
          const edm4hep::SimTrackerHitCollection&,
          const edm4hep::TrackerHitSimTrackerHitLinkCollection&,
          const edm4hep::SimTrackerHitCollection&)> {
public:
  PixelHitBDTTestAlg(const std::string& name, ISvcLocator* svcLoc);

  StatusCode initialize() final;
  std::tuple<edm4hep::TrackerHitPlaneCollection, podio::UserDataCollection<float>, podio::UserDataCollection<float>>
  operator()(const edm4hep::TrackerHitPlaneCollection& hits,
             const edm4hep::TrackerHitSimTrackerHitLinkCollection& rawHitRelations,
             const edm4hep::SimTrackerHitCollection& rawSimHits,
             const edm4hep::TrackerHitSimTrackerHitLinkCollection& truthRelations,
             const edm4hep::SimTrackerHitCollection& truthSimHits) const final;

private:
  static constexpr std::size_t kNVars = 33;
  static constexpr std::size_t kNPixelVars = 9;
  static const std::array<const char*, kNVars> s_varNames;

  void bindReaderVariables(TMVA::Reader& reader, std::array<float, kNVars>& vars) const;
  std::array<float, kNVars> buildFeatures(const edm4hep::TrackerHitPlane& hit,
                                          const std::vector<edm4hep::SimTrackerHit>& rawHits) const;

  static std::pair<float, float> getClusterSize(const std::vector<edm4hep::SimTrackerHit>& rawHits);
  static float getRMS(float clusterPos, const std::vector<edm4hep::SimTrackerHit>& rawHits, bool useX);
  static float getCov(float clusterPosX, float clusterPosY, const std::vector<edm4hep::SimTrackerHit>& rawHits);
  static float getSkew(float clusterPos, const std::vector<edm4hep::SimTrackerHit>& rawHits, bool useX);

  Gaudi::Property<std::string> m_weightsFile{
      this, "WeightsFile", "TMVAClassification_BDT.weights.xml", "Path to TMVA XML weights file"};
  Gaudi::Property<std::string> m_methodName{
    this, "MethodName", "BDT", "TMVA method name inside weights file"};
  Gaudi::Property<float> m_minBdtScore{
    this, "MinBdtScore", 0.0f, "Minimum BDT score required to keep a hit in OutputScoredHitCollection"};
  Gaudi::Property<int> m_numThreads{
    this, "NumThreads", 1, "Number of threads for per-hit BDT evaluation"};
};