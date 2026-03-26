#include "PixelHitBDTAlg.h"

#include <edm4hep/TrackerHit.h>
#include <edm4hep/TrackerHitPlane.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

DECLARE_COMPONENT(PixelHitBDTAlg)

namespace {
enum FeatureIndex : std::size_t {
  kClusterArrivalTime = 0,
  kClusterEnergyDeposited,
  kIncidentAngle,
  kClusterSizeX,
  kClusterSizeY,
  kClusterSizeTot,
  kClusterX,
  kClusterY,
  kClusterZ,
  kClusterRmsX,
  kClusterRmsY,
  kClusterSkewX,
  kClusterSkewY,
  kClusterAspect,
  kClusterEcc,
  kPixelEnergy0,
  kPixelTime0,
  kPixelEnergy1,
  kPixelTime1,
  kPixelEnergy2,
  kPixelTime2,
  kPixelEnergy3,
  kPixelTime3,
  kPixelEnergy4,
  kPixelTime4,
  kPixelEnergy5,
  kPixelTime5,
  kPixelEnergy6,
  kPixelTime6,
  kPixelEnergy7,
  kPixelTime7,
  kPixelEnergy8,
  kPixelTime8
};
}

const std::array<const char*, PixelHitBDTAlg::kNVars> PixelHitBDTAlg::s_varNames = {
    "Cluster_ArrivalTime",         "Cluster_EnergyDeposited",      "Incident_Angle",
    "Cluster_Size_x",              "Cluster_Size_y",               "Cluster_Size_tot",
    "Cluster_x",                   "Cluster_y",                    "Cluster_z",
    "Cluster_RMS_x",               "Cluster_RMS_y",                "Cluster_Skew_x",
    "Cluster_Skew_y",              "Cluster_AspectRatio",          "Cluster_Eccentricity",
    "PixelHits_EnergyDeposited_0", "PixelHits_ArrivalTime_0",      "PixelHits_EnergyDeposited_1",
    "PixelHits_ArrivalTime_1",     "PixelHits_EnergyDeposited_2",  "PixelHits_ArrivalTime_2",
    "PixelHits_EnergyDeposited_3", "PixelHits_ArrivalTime_3",      "PixelHits_EnergyDeposited_4",
    "PixelHits_ArrivalTime_4",     "PixelHits_EnergyDeposited_5",  "PixelHits_ArrivalTime_5",
    "PixelHits_EnergyDeposited_6", "PixelHits_ArrivalTime_6",      "PixelHits_EnergyDeposited_7",
    "PixelHits_ArrivalTime_7",     "PixelHits_EnergyDeposited_8",  "PixelHits_ArrivalTime_8"};

PixelHitBDTAlg::PixelHitBDTAlg(const std::string& name, ISvcLocator* svcLoc)
  : MultiTransformer(name, svcLoc,
             {KeyValues("InputHitCollectionName", {"VXDBarrelHits"}),
            KeyValues("InputRawHitRelationCollectionName", {"VTXRawHitRelations"}),
            KeyValues("InputSimHitCollectionName", {"VertexBarrel"})},
             {KeyValues("OutputFilteredHitCollectionName", {"VXDBarrelHitsBDT"})}) {}

StatusCode PixelHitBDTAlg::initialize() {
  if (!std::filesystem::exists(m_weightsFile.value())) {
    error() << "Weights file not found: " << m_weightsFile.value() << endmsg;
    return StatusCode::FAILURE;
  }

  if (m_numThreads.value() < 1) {
    warning() << "NumThreads must be >= 1. Forcing NumThreads = 1." << endmsg;
    m_numThreads = 1;
  }

  auto reader = std::make_unique<TMVA::Reader>("!Color:!Silent");
  std::array<float, kNVars> initVars{};
  bindReaderVariables(*reader, initVars);

  try {
    reader->BookMVA(m_methodName.value(), m_weightsFile.value());
  } catch (const std::exception& ex) {
    error() << "Failed to initialize TMVA::Reader from weights file '" << m_weightsFile.value() << "': " << ex.what()
            << endmsg;
    return StatusCode::FAILURE;
  }

  info() << "Initialized PixelHitBDTAlg with weights file: " << m_weightsFile.value() << endmsg;
  return StatusCode::SUCCESS;
}

std::tuple<edm4hep::TrackerHitPlaneCollection> PixelHitBDTAlg::operator()(
    const edm4hep::TrackerHitPlaneCollection& inputHits,
    const edm4hep::TrackerHitSimTrackerHitLinkCollection& rawHitRelations,
    const edm4hep::SimTrackerHitCollection& simHits) const {
  using HitKey = std::pair<uint32_t, int32_t>;
  using RelRange = std::pair<std::size_t, std::size_t>;

  const auto makeKey = [](const podio::ObjectID& id) { return HitKey{id.collectionID, id.index}; };

  const std::size_t nHits = inputHits.size();
  std::vector<RelRange> relationRanges(nHits, {0, 0});

  // Phase 1: linear relation scan with cursor to find contiguous relation block per hit.
  std::size_t relationCursor = 0;
  for (std::size_t iHit = 0; iHit < nHits; ++iHit) {
    const HitKey hitKey = makeKey(inputHits[iHit].id());

    while (relationCursor < rawHitRelations.size()) {
      const HitKey relFromKey = makeKey(rawHitRelations[relationCursor].getFrom().id());
      if (relFromKey < hitKey) {
        ++relationCursor;
        continue;
      }
      break;
    }

    const std::size_t begin = relationCursor;
    while (relationCursor < rawHitRelations.size() && makeKey(rawHitRelations[relationCursor].getFrom().id()) == hitKey) {
      ++relationCursor;
    }
    relationRanges[iHit] = {begin, relationCursor};
  }

  const std::size_t nReaderSlots = static_cast<std::size_t>(std::max(1, m_numThreads.value()));
  std::vector<std::unique_ptr<TMVA::Reader>> readers;
  std::vector<std::array<float, kNVars>> readerVars(nReaderSlots);
  readers.reserve(nReaderSlots);

  for (std::size_t i = 0; i < nReaderSlots; ++i) {
    auto slotReader = std::make_unique<TMVA::Reader>("!Color:!Silent");
    bindReaderVariables(*slotReader, readerVars[i]);
    slotReader->BookMVA(m_methodName.value(), m_weightsFile.value());
    readers.push_back(std::move(slotReader));
  }

  std::vector<uint8_t> passMask(nHits, 0);

  // Give each TBB worker its own reader and range of hits to process, and combine results at the end.
  auto evaluateRange = [&](const tbb::blocked_range<std::size_t>& r) {
    int slot = tbb::this_task_arena::current_thread_index();
    if (slot < 0 || static_cast<std::size_t>(slot) >= nReaderSlots) {
      slot = 0;
    }

    TMVA::Reader& localReader = *readers[static_cast<std::size_t>(slot)];
    std::array<float, kNVars>& localVars = readerVars[static_cast<std::size_t>(slot)];

    std::vector<edm4hep::SimTrackerHit> rawHits;
    rawHits.reserve(16);

    // Loop over assigned hits, build features, evaluate BDT, and fill pass mask.
    for (std::size_t iHit = r.begin(); iHit != r.end(); ++iHit) {
      rawHits.clear();

      const auto [relBegin, relEnd] = relationRanges[iHit];
      for (std::size_t iRel = relBegin; iRel < relEnd; ++iRel) {
        const auto& rel = rawHitRelations[iRel];
        const auto toID = rel.getTo().id();
        if (toID.index < 0) {
          continue;
        }

        edm4hep::SimTrackerHit simHit;
        if (toID.collectionID == simHits.getID() && static_cast<size_t>(toID.index) < simHits.size()) {
          simHit = simHits[static_cast<size_t>(toID.index)];
        } else if (rel.getTo().isAvailable()) {
          simHit = rel.getTo();
        }

        if (simHit.isAvailable()) {
          rawHits.push_back(simHit);
        }
      }

      localVars = buildFeatures(inputHits[iHit], rawHits);
      const float score = static_cast<float>(localReader.EvaluateMVA(m_methodName.value()));
      passMask[iHit]    = static_cast<uint8_t>(score >= m_minBdtScore.value());
    }
  };

  // Run in parallel if more than one thread is requested
  if (nReaderSlots > 1) {
    tbb::task_arena arena(static_cast<int>(nReaderSlots));
    arena.execute([&] { tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nHits), evaluateRange); });
  } else {
    evaluateRange(tbb::blocked_range<std::size_t>(0, nHits));
  }

  edm4hep::TrackerHitPlaneCollection outFiltered;
  outFiltered.setSubsetCollection();

  // Push hits that pass the BDT score cut into the output collection.
  for (std::size_t iHit = 0; iHit < nHits; ++iHit) {
    if (passMask[iHit] != 0) {
      outFiltered.push_back(inputHits[iHit]);
    }
  }

  debug() << "Input hits: " << inputHits.size() << ", selected: " << outFiltered.size()
          << ", raw relations: " << rawHitRelations.size()
          << ", sim hits: " << simHits.size()
          << ", NumThreads: " << nReaderSlots
          << ", MinBdtScore: " << m_minBdtScore.value() << endmsg;

    return std::make_tuple(std::move(outFiltered));
}

void PixelHitBDTAlg::bindReaderVariables(TMVA::Reader& reader, std::array<float, kNVars>& vars) const {
  for (std::size_t i = 0; i < kNVars; ++i) {
    reader.AddVariable(s_varNames[i], &vars[i]);
  }
}

std::pair<float, float> PixelHitBDTAlg::getClusterSize(const std::vector<edm4hep::SimTrackerHit>& rawHits) {
  if (rawHits.empty()) {
    return {0.0f, 0.0f};
  }

  float xmin = rawHits.front().getPosition().x;
  float xmax = rawHits.front().getPosition().x;
  float ymin = rawHits.front().getPosition().y;
  float ymax = rawHits.front().getPosition().y;

  for (const auto& h : rawHits) {
    const auto& p = h.getPosition();
    xmin          = std::min(xmin, static_cast<float>(p.x));
    xmax          = std::max(xmax, static_cast<float>(p.x));
    ymin          = std::min(ymin, static_cast<float>(p.y));
    ymax          = std::max(ymax, static_cast<float>(p.y));
  }

  return {(xmax - xmin) + 1.0f, (ymax - ymin) + 1.0f};
}

float PixelHitBDTAlg::getRMS(
    float clusterPos,
    const std::vector<edm4hep::SimTrackerHit>& rawHits,
    bool useX) {
  if (rawHits.empty()) {
    return 0.0f;
  }

  double sumNum = 0.0;
  double sumDen = 0.0;
  for (const auto& h : rawHits) {
    const float local = useX ? static_cast<float>(h.getPosition().x) : static_cast<float>(h.getPosition().y);
    const float eDep  = h.getEDep();
    sumNum += static_cast<double>(eDep) * static_cast<double>(clusterPos - local) * static_cast<double>(clusterPos - local);
    sumDen += static_cast<double>(eDep);
  }

  return (sumDen > 0.0) ? static_cast<float>(sumNum / sumDen) : 0.0f; 
}

float PixelHitBDTAlg::getCov(
    float clusterPosX,
    float clusterPosY,
    const std::vector<edm4hep::SimTrackerHit>& rawHits) {
  if (rawHits.empty()) {
    return 0.0f;
  }

  double sumNum = 0.0;
  double sumDen = 0.0;
  for (const auto& h : rawHits) {
    const auto& p = h.getPosition();
    const float lx = static_cast<float>(p.x);
    const float ly = static_cast<float>(p.y);
    const float e  = h.getEDep();
    sumNum += static_cast<double>(e) * static_cast<double>(clusterPosX - lx) * static_cast<double>(clusterPosY - ly);
    sumDen += static_cast<double>(e);
  }

  return (sumDen > 0.0) ? static_cast<float>(sumNum / sumDen) : 0.0f;
}

float PixelHitBDTAlg::getSkew(
    float clusterPos,
    const std::vector<edm4hep::SimTrackerHit>& rawHits,
    bool useX) {
  if (rawHits.empty()) {
    return 0.0f;
  }

  const float rms = getRMS(clusterPos, rawHits, useX);
  if (rms <= 0.0f) {
    return 0.0f;
  }

  const double rmsTerm = std::pow(std::sqrt(static_cast<double>(rms)), 3.0);
  if (rmsTerm <= 0.0) {
    return 0.0f;
  }

  double sumNum = 0.0;
  double sumDen = 0.0;
  for (const auto& h : rawHits) {
    const float local = useX ? static_cast<float>(h.getPosition().x) : static_cast<float>(h.getPosition().y);
    const float eDep  = h.getEDep();
    const double diff = static_cast<double>(clusterPos - local);
    sumNum += static_cast<double>(eDep) * diff * diff * diff;
    sumDen += static_cast<double>(eDep) * rmsTerm;
  }

  return (sumDen > 0.0) ? static_cast<float>(sumNum / sumDen) : 0.0f; 
}

std::array<float, PixelHitBDTAlg::kNVars> PixelHitBDTAlg::buildFeatures(
    const edm4hep::TrackerHitPlane& hit,
    const std::vector<edm4hep::SimTrackerHit>& rawHits) const {
  std::array<float, kNVars> features{};
  features.fill(0.0f);

  const auto& pos = hit.getPosition();
  const float x   = static_cast<float>(pos.x);
  const float y   = static_cast<float>(pos.y);
  const float z   = static_cast<float>(pos.z);
  const float r   = std::sqrt(x * x + y * y);

  // Directly available hit-level quantities.
  features[kClusterArrivalTime]     = hit.getTime();
  features[kClusterEnergyDeposited] = hit.getEDep();
  features[kIncidentAngle]          = std::atan2(r, z);
  features[kClusterX]               = x;
  features[kClusterY]               = y;
  features[kClusterZ]               = z;

  const auto [sizeX, sizeY] = getClusterSize(rawHits);
  features[kClusterSizeX]   = sizeX;
  features[kClusterSizeY]   = sizeY;
  features[kClusterSizeTot] = static_cast<float>(rawHits.size());

  features[kClusterRmsX]  = getRMS(x, rawHits, true);
  features[kClusterRmsY]  = getRMS(y, rawHits, false);
  features[kClusterSkewX] = getSkew(x, rawHits, true);
  features[kClusterSkewY] = getSkew(y, rawHits, false);

  const float clusterXY = getCov(x, y, rawHits);
  const float trace     = features[kClusterRmsX] + features[kClusterRmsY];
  const float det       = features[kClusterRmsX] * features[kClusterRmsY] - clusterXY * clusterXY;
  const float disc      = std::max(0.0f, trace * trace - 4.0f * det);
  const float sqrtDisc  = std::sqrt(disc);
  const float lambda1   = 0.5f * (trace + sqrtDisc);
  const float lambda2   = 0.5f * (trace - sqrtDisc);

  if (lambda1 >= lambda2) {
    features[kClusterAspect] = (lambda2 > 0.0f) ? std::sqrt(lambda1 / lambda2) : 0.0f;
    features[kClusterEcc] = (lambda1 > 0.0f) ? std::sqrt(std::max(0.0f, 1.0f - (lambda2 / lambda1))) : 0.0f;
  } else {
    features[kClusterAspect] = (lambda1 > 0.0f) ? std::sqrt(lambda2 / lambda1) : 0.0f;
    features[kClusterEcc] = (lambda2 > 0.0f) ? std::sqrt(std::max(0.0f, 1.0f - (lambda1 / lambda2))) : 0.0f;
  }

  std::vector<std::pair<float, float>> pixelETime;
  pixelETime.reserve(rawHits.size());
  for (const auto& raw : rawHits) {
    pixelETime.emplace_back(raw.getEDep(), raw.getTime());
  }

  std::sort(pixelETime.begin(), pixelETime.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

  const std::size_t nFill = std::min(kNPixelVars, pixelETime.size());
  for (std::size_t i = 0; i < nFill; ++i) {
    features[kPixelEnergy0 + 2 * i] = pixelETime[i].first;
    features[kPixelTime0 + 2 * i]   = pixelETime[i].second;
  }

  return features;
}
