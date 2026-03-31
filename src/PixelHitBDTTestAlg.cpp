#include "PixelHitBDTTestAlg.h"

#include <edm4hep/TrackerHit.h>
#include <edm4hep/TrackerHitPlane.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

DECLARE_COMPONENT(PixelHitBDTTestAlg)

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

const std::array<const char*, PixelHitBDTTestAlg::kNVars> PixelHitBDTTestAlg::s_varNames = {
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

PixelHitBDTTestAlg::PixelHitBDTTestAlg(const std::string& name, ISvcLocator* svcLoc)
    : MultiTransformer(name, svcLoc,
                       {KeyValues("InputHitCollectionName", {"VXDBarrelHits"}),
                        KeyValues("InputRawHitRelationCollectionName", {"VTXRawHitRelations"}),
                        KeyValues("InputRawSimHitCollectionName", {"VertexBarrel"}),
                        KeyValues("InputTruthRelationCollectionName", {"VXDBarrelHitTruthRelations"}),
                        KeyValues("InputTruthSimHitCollectionName", {"VertexBarrel"})},
                       {KeyValues("OutputScoredHitCollectionName", {"VXDBarrelHitsScored"}),
                        KeyValues("OutputBdtScoreCollectionName", {"VXDBarrelHitBDTScores"}),
                        KeyValues("OutputTruthLabelCollectionName", {"VXDBarrelHitTruthLabels"})}) {}

StatusCode PixelHitBDTTestAlg::initialize() {
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
    error() << "Failed to initialize TMVA::Reader from weights file '" << m_weightsFile.value() << "': "
            << ex.what() << endmsg;
    return StatusCode::FAILURE;
  }

  info() << "Initialized PixelHitBDTTestAlg with weights file: " << m_weightsFile.value() << endmsg;
  return StatusCode::SUCCESS;
}

std::tuple<edm4hep::TrackerHitPlaneCollection, podio::UserDataCollection<float>, podio::UserDataCollection<float>>
PixelHitBDTTestAlg::operator()(const edm4hep::TrackerHitPlaneCollection& inputHits,
                               const edm4hep::TrackerHitSimTrackerHitLinkCollection& rawHitRelations,
                               const edm4hep::SimTrackerHitCollection& rawSimHits,
                               const edm4hep::TrackerHitSimTrackerHitLinkCollection& truthRelations,
                               const edm4hep::SimTrackerHitCollection& truthSimHits) const {
  using HitKey = std::pair<uint32_t, int32_t>;
  using RelRange = std::pair<std::size_t, std::size_t>;

  const auto makeKey = [](const podio::ObjectID& id) { return HitKey{id.collectionID, id.index}; };

  const auto buildRelationRanges = [&](const edm4hep::TrackerHitSimTrackerHitLinkCollection& relations) {
    std::vector<RelRange> relationRanges(inputHits.size(), {0, 0});
    std::size_t relationCursor = 0;
    for (std::size_t iHit = 0; iHit < inputHits.size(); ++iHit) {
      const HitKey hitKey = makeKey(inputHits[iHit].id());

      while (relationCursor < relations.size()) {
        const HitKey relFromKey = makeKey(relations[relationCursor].getFrom().id());
        if (relFromKey < hitKey) {
          ++relationCursor;
          continue;
        }
        break;
      }

      const std::size_t begin = relationCursor;
      while (relationCursor < relations.size() && makeKey(relations[relationCursor].getFrom().id()) == hitKey) {
        ++relationCursor;
      }
      relationRanges[iHit] = {begin, relationCursor};
    }
    return relationRanges;
  };

  const std::size_t nHits = inputHits.size();
  const auto rawRanges = buildRelationRanges(rawHitRelations);

  // Truth matching is expected to be 1:1 with contiguous hit IDs [0, nHits).
  std::vector<float> truthLabels(nHits, 0.0f);
  for (const auto& rel : truthRelations) {
    const auto fromID = rel.getFrom().id();
    if (fromID.collectionID != inputHits.getID() || fromID.index < 0) {
      continue;
    }

    const std::size_t iHit = static_cast<std::size_t>(fromID.index);
    if (iHit >= nHits) {
      continue;
    }

    const auto toID = rel.getTo().id();
    bool isOverlay = true;
    if (toID.index >= 0 && toID.collectionID == truthSimHits.getID() && static_cast<std::size_t>(toID.index) < truthSimHits.size()) {
      isOverlay = truthSimHits[static_cast<std::size_t>(toID.index)].isOverlay();
    } else if (rel.getTo().isAvailable()) {
      isOverlay = rel.getTo().isOverlay();
    }

    // Label is 1 for non-overlay truth (signal-like), 0 for overlay/BIB.
    if (!isOverlay) {
      truthLabels[iHit] = 1.0f;
    }
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

  std::vector<float> bdtScores(nHits, 0.0f);

  auto evaluateRange = [&](const tbb::blocked_range<std::size_t>& r) {
    int slot = tbb::this_task_arena::current_thread_index();
    if (slot < 0 || static_cast<std::size_t>(slot) >= nReaderSlots) {
      slot = 0;
    }

    TMVA::Reader& localReader = *readers[static_cast<std::size_t>(slot)];
    std::array<float, kNVars>& localVars = readerVars[static_cast<std::size_t>(slot)];

    std::vector<edm4hep::SimTrackerHit> rawHits;
    rawHits.reserve(16);

    for (std::size_t iHit = r.begin(); iHit != r.end(); ++iHit) {
      rawHits.clear();

      const auto [rawBegin, rawEnd] = rawRanges[iHit];
      for (std::size_t iRel = rawBegin; iRel < rawEnd; ++iRel) {
        const auto& rel = rawHitRelations[iRel];
        const auto toID = rel.getTo().id();
        if (toID.index < 0) {
          continue;
        }

        edm4hep::SimTrackerHit simHit;
        if (toID.collectionID == rawSimHits.getID() && static_cast<std::size_t>(toID.index) < rawSimHits.size()) {
          simHit = rawSimHits[static_cast<std::size_t>(toID.index)];
        } else if (rel.getTo().isAvailable()) {
          simHit = rel.getTo();
        }

        if (simHit.isAvailable()) {
          rawHits.push_back(simHit);
        }
      }

      localVars = buildFeatures(inputHits[iHit], rawHits);
      bdtScores[iHit] = static_cast<float>(localReader.EvaluateMVA(m_methodName.value()));

    }
  };

  if (nReaderSlots > 1) {
    tbb::task_arena arena(static_cast<int>(nReaderSlots));
    arena.execute([&] { tbb::parallel_for(tbb::blocked_range<std::size_t>(0, nHits), evaluateRange); });
  } else {
    evaluateRange(tbb::blocked_range<std::size_t>(0, nHits));
  }

  edm4hep::TrackerHitPlaneCollection outHits;
  outHits.setSubsetCollection();
  podio::UserDataCollection<float> outScores;
  podio::UserDataCollection<float> outLabels;

  for (std::size_t iHit = 0; iHit < nHits; ++iHit) {
    if (bdtScores[iHit] >= m_minBdtScore.value()) {
      outHits.push_back(inputHits[iHit]);
    }
    outScores.push_back(bdtScores[iHit]);
    outLabels.push_back(truthLabels[iHit]);
  }

  debug() << "Input hits: " << inputHits.size() << ", selected hits: " << outHits.size()
          << ", raw relations: " << rawHitRelations.size() << ", truth relations: " << truthRelations.size()
          << ", NumThreads: " << nReaderSlots << ", MinBdtScore: " << m_minBdtScore.value() << endmsg;

  return std::make_tuple(std::move(outHits), std::move(outScores), std::move(outLabels));
}

void PixelHitBDTTestAlg::bindReaderVariables(TMVA::Reader& reader, std::array<float, kNVars>& vars) const {
  for (std::size_t i = 0; i < kNVars; ++i) {
    reader.AddVariable(s_varNames[i], &vars[i]);
  }
}

std::pair<float, float> PixelHitBDTTestAlg::getClusterSize(const std::vector<edm4hep::SimTrackerHit>& rawHits) {
  if (rawHits.empty()) {
    return {0.0f, 0.0f};
  }

  float xmin = rawHits.front().getPosition().x;
  float xmax = rawHits.front().getPosition().x;
  float ymin = rawHits.front().getPosition().y;
  float ymax = rawHits.front().getPosition().y;

  for (const auto& h : rawHits) {
    const auto& p = h.getPosition();
    xmin = std::min(xmin, static_cast<float>(p.x));
    xmax = std::max(xmax, static_cast<float>(p.x));
    ymin = std::min(ymin, static_cast<float>(p.y));
    ymax = std::max(ymax, static_cast<float>(p.y));
  }

  return {(xmax - xmin) + 1.0f, (ymax - ymin) + 1.0f};
}

float PixelHitBDTTestAlg::getRMS(float clusterPos, const std::vector<edm4hep::SimTrackerHit>& rawHits, bool useX) {
  if (rawHits.empty()) {
    return 0.0f;
  }

  double sumNum = 0.0;
  double sumDen = 0.0;
  for (const auto& h : rawHits) {
    const float local = useX ? static_cast<float>(h.getPosition().x) : static_cast<float>(h.getPosition().y);
    const float eDep = h.getEDep();
    sumNum += static_cast<double>(eDep) * static_cast<double>(clusterPos - local) * static_cast<double>(clusterPos - local);
    sumDen += static_cast<double>(eDep);
  }

  return (sumDen > 0.0) ? static_cast<float>(sumNum / sumDen) : 0.0f;
}

float PixelHitBDTTestAlg::getCov(float clusterPosX,
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
    const float e = h.getEDep();
    sumNum += static_cast<double>(e) * static_cast<double>(clusterPosX - lx) * static_cast<double>(clusterPosY - ly);
    sumDen += static_cast<double>(e);
  }

  return (sumDen > 0.0) ? static_cast<float>(sumNum / sumDen) : 0.0f;
}

float PixelHitBDTTestAlg::getSkew(float clusterPos, const std::vector<edm4hep::SimTrackerHit>& rawHits, bool useX) {
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
    const float eDep = h.getEDep();
    const double diff = static_cast<double>(clusterPos - local);
    sumNum += static_cast<double>(eDep) * diff * diff * diff;
    sumDen += static_cast<double>(eDep) * rmsTerm;
  }

  return (sumDen > 0.0) ? static_cast<float>(sumNum / sumDen) : 0.0f;
}

std::array<float, PixelHitBDTTestAlg::kNVars> PixelHitBDTTestAlg::buildFeatures(
    const edm4hep::TrackerHitPlane& hit,
    const std::vector<edm4hep::SimTrackerHit>& rawHits) const {
  std::array<float, kNVars> features{};
  features.fill(0.0f);

  const auto& pos = hit.getPosition();
  const float x = static_cast<float>(pos.x);
  const float y = static_cast<float>(pos.y);
  const float z = static_cast<float>(pos.z);
  const float r = std::sqrt(x * x + y * y);

  features[kClusterArrivalTime] = hit.getTime();
  features[kClusterEnergyDeposited] = hit.getEDep();
  features[kIncidentAngle] = std::atan2(r, z);
  features[kClusterX] = x;
  features[kClusterY] = y;
  features[kClusterZ] = z;

  const auto [sizeX, sizeY] = getClusterSize(rawHits);
  features[kClusterSizeX] = sizeX;
  features[kClusterSizeY] = sizeY;
  features[kClusterSizeTot] = static_cast<float>(rawHits.size());

  features[kClusterRmsX] = getRMS(x, rawHits, true);
  features[kClusterRmsY] = getRMS(y, rawHits, false);
  features[kClusterSkewX] = getSkew(x, rawHits, true);
  features[kClusterSkewY] = getSkew(y, rawHits, false);

  const float clusterXY = getCov(x, y, rawHits);
  const float trace = features[kClusterRmsX] + features[kClusterRmsY];
  const float det = features[kClusterRmsX] * features[kClusterRmsY] - clusterXY * clusterXY;
  const float disc = std::max(0.0f, trace * trace - 4.0f * det);
  const float sqrtDisc = std::sqrt(disc);
  const float lambda1 = 0.5f * (trace + sqrtDisc);
  const float lambda2 = 0.5f * (trace - sqrtDisc);

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
    features[kPixelTime0 + 2 * i] = pixelETime[i].second;
  }

  return features;
}