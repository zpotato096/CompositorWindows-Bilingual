#pragma once

#include "filters/LightingFilters.h"
#include <QString>

namespace compositor::ui {

using LightingAnalysis=std::function<std::shared_ptr<const filters::LightingScene>(const std::function<bool()>&)>;
LightingAnalysis prepareLightingAnalysis(std::shared_ptr<const Raster> source,const QString& modelDirectory,bool depth);

}
