#include "ProjectSaveJob.h"
#include "persistence/ProjectStore.h"
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrentRun>
#include <filesystem>

namespace compositor::ui {
struct ProjectSaveJob::Impl {
    Host host;
    using Result=std::pair<QString,std::string>;
    QFutureWatcher<Result> watcher;
};

ProjectSaveJob::ProjectSaveJob(Document document,std::string activeLayer,QString path,Host host,QObject* parent,bool macCompatible)
    :QObject(parent),impl_(std::make_unique<Impl>()){
    setObjectName("projectSaveJob");
    impl_->host=std::move(host);
    connect(&impl_->watcher,&QFutureWatcher<Impl::Result>::finished,this,[this]{
        auto [error,fingerprint]=impl_->watcher.result();
        auto completed=impl_->host.completed;
        completed(error,std::move(fingerprint));
        deleteLater();
    });
    // The worker receives only value copies. Document raster tiles are shared and immutable.
    impl_->watcher.setFuture(QtConcurrent::run([document=std::move(document),activeLayer=std::move(activeLayer),path=std::move(path),beforeWrite=impl_->host.beforeWrite,macCompatible]() -> Impl::Result {
        try{
            if(beforeWrite)beforeWrite();
            ProjectStore store(makeWicProjectCodec());
            if(macCompatible){store.saveMacCompatible(std::filesystem::path(path.toStdWString()),document,activeLayer);return {};}
            return {{},store.saveObserved(std::filesystem::path(path.toStdWString()),document,activeLayer)};
        }catch(const std::exception& error){
            return {QString::fromUtf8(error.what()),{}};
        }
    }));
}

ProjectSaveJob::~ProjectSaveJob(){
    disconnect(&impl_->watcher,nullptr,this,nullptr);
    impl_->watcher.waitForFinished();
}
}
