cmake_minimum_required(VERSION 3.18)

project(ML4FPSGaudi LANGUAGES CXX)

find_package(Gaudi REQUIRED)
find_package(k4FWCore REQUIRED)
find_package(EDM4HEP REQUIRED)
find_package(ROOT REQUIRED COMPONENTS TMVA)
find_package(TBB REQUIRED)

gaudi_add_module(ML4FPSGaudiPlugins
  SOURCES
    src/PixelHitBDTAlg.cpp
    src/PixelHitBDTTestAlg.cpp
  LINK
    Gaudi::GaudiKernel
    k4FWCore::k4FWCore
    EDM4HEP::edm4hep
    ROOT::TMVA
    TBB::tbb
)

target_include_directories(ML4FPSGaudiPlugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)
