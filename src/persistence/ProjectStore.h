#pragma once
#include "core/Document.h"
#include <filesystem>
#include <functional>
#include <string>

namespace compositor {
// Color assets are PNG, or single-image HDR EXR from v13. Masks stay 8-bit
// gray PNG without alpha. Each decoder checks dimensions before allocating.
struct ProjectAssetCodec {
    std::function<std::shared_ptr<const Raster>(const std::filesystem::path&)> readColor;
    std::function<std::shared_ptr<const GrayRaster>(const std::filesystem::path&)> readGray;
    std::function<void(const std::filesystem::path&,const Raster&)> writeColor;
    std::function<void(const std::filesystem::path&,const GrayRaster&)> writeGray;
    std::function<std::shared_ptr<const DeepImage>(const std::filesystem::path&)> readDeep;
    std::function<void(const std::filesystem::path&,const DeepImage&)> writeDeep;
};
struct OpenProject { Document document; std::string activeLayer; int readVersion{7}; };
struct ObservedProject {
    std::string fingerprint;
    std::optional<OpenProject> project;
};
enum class SaveFaultPoint { AfterStage, AfterJournal, AfterBackup, AfterInstall, BeforeCleanup, BeforeJournalReplace, BeforeMetadataTempCreate };
using SaveFaultInjector=std::function<void(SaveFaultPoint)>;
// Non-null only while a BeforeMetadataTempCreate hook is running.
const std::filesystem::path* metadataTempAwaitingCreate();
class IProjectStore {
public:
    virtual ~IProjectStore()=default;
    virtual OpenProject load(const std::filesystem::path&)=0;
    virtual void save(const std::filesystem::path&,const Document&,const std::string& activeLayer)=0;
    virtual void recover(const std::filesystem::path&)=0;
};
// Directory replacement is recoverable with a sibling journal, not claimed atomic.
class ProjectStore final:public IProjectStore {
    ProjectAssetCodec codec_;
    SaveFaultInjector fault_;
    OpenProject loadPackage(const std::filesystem::path&) const;
    void recoverLocked(const std::filesystem::path&);
    std::string savePackage(const std::filesystem::path&,const Document&,const std::string&,bool fingerprint,bool macCompatible=false);
public:
    explicit ProjectStore(ProjectAssetCodec codec,SaveFaultInjector fault={});
    OpenProject load(const std::filesystem::path&) override;
    ObservedProject openObserved(const std::filesystem::path&);
    // Watches read committed data without modifying a foreign transaction or attempting recovery.
    ObservedProject observe(const std::filesystem::path&,const std::string& known) const;
    void save(const std::filesystem::path&,const Document&,const std::string& activeLayer) override;
    std::string saveObserved(const std::filesystem::path&,const Document&,const std::string& activeLayer);
    void saveMacCompatible(const std::filesystem::path&,const Document&,const std::string& activeLayer);
    void recover(const std::filesystem::path&) override;
};
// Production WIC adapter. Defined in WicProjectCodec.cpp.
ProjectAssetCodec makeWicProjectCodec();
}
