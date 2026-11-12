#pragma once

#include "core/Document.h"
#include <array>
#include <functional>

namespace compositor::filters {

struct RimLamp {
    double angle{135},strength{50},width{30},choke{};
    std::array<double,3> color{1,.86,.7};
};
struct RimLightingSettings {
    double glow{10};
    std::array<RimLamp,3> lights{{{}, {-45,50,30,0,{.7,.84,1}}, {135,50,30,0,{1,1,1}}}};
    int count{1};
};
struct VolumetricLightingSettings {
    double haze{60},distance{30},depthContrast{},size{20},intensity{30},length{},shafts{};
    bool keepsSubjectClear{true};
    std::optional<Point> center;
    std::array<double,3> color{1,.86,.68};
};
struct LightingScene {
    std::shared_ptr<const Raster> sample;
    std::vector<float> subject,depth;
    std::string depthIssue;
};

struct Request;
struct Result;
bool isLighting(const Request&);
bool lightingIsIdentity(const Request&);
Result applyLighting(const Request&,Result);
std::shared_ptr<const Raster> extractLightingDepth(const Request&);

}
