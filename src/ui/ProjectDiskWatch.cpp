#include "ProjectDiskWatch.h"
#include "imaging/win32_file_path.h"
#include <QDir>
#include <QFileInfo>
#include <QtConcurrent/QtConcurrentRun>
#include <utility>
#include <Windows.h>

namespace compositor::ui {

ProjectDiskWatch::ProjectDiskWatch(QString path,std::string fingerprint,std::function<bool()> ready,
                                   std::function<void(OpenProject)> changed,QObject* parent)
    : QObject(parent),path_(std::move(path)),fingerprint_(std::move(fingerprint)),ready_(std::move(ready)),changed_(std::move(changed)){
    timer_.setSingleShot(true);
    connect(&timer_,&QTimer::timeout,this,[this]{check();});
    connect(&filesystem_,&QFileSystemWatcher::directoryChanged,this,[this]{noteChange();});
    connect(&filesystem_,&QFileSystemWatcher::fileChanged,this,[this]{noteChange();});
    connect(&imageWrites_,&QWinEventNotifier::activated,this,[this](HANDLE handle){
        if(!FindNextChangeNotification(handle))closeImageWatch();
        noteChange();
    });
    connect(&reader_,&QFutureWatcher<std::optional<ObservedProject>>::finished,this,[this]{finish();});
    arm();
    noteChange();
}

ProjectDiskWatch::~ProjectDiskWatch(){closeImageWatch();}

void ProjectDiskWatch::closeImageWatch(){
    const auto handle=imageWrites_.handle();
    imageWrites_.setEnabled(false);
    imageWrites_.setHandle(nullptr);
    if(handle&&handle!=INVALID_HANDLE_VALUE)FindCloseChangeNotification(handle);
}

void ProjectDiskWatch::arm(){
    const auto files=filesystem_.files()+filesystem_.directories();
    if(!files.isEmpty())filesystem_.removePaths(files);
    const QDir folder(path_);
    QStringList paths{QFileInfo(path_).absolutePath(),path_,folder.filePath("manifest.json"),folder.filePath("images"),folder.filePath("cmyk.icc")};
    for(const auto& path:paths)if(QFileInfo::exists(path))filesystem_.addPath(path);
    // Qt's Windows directory watcher doesn't request FILE_NOTIFY_CHANGE_LAST_WRITE.
    // Watch the image directory once at the Win32 level for in-place asset edits.
    closeImageWatch();
    const auto images=imaging::win32FilePath(std::filesystem::path(folder.filePath("images").toStdWString()));
    const auto event=FindFirstChangeNotificationW(images.c_str(),FALSE,FILE_NOTIFY_CHANGE_LAST_WRITE|FILE_NOTIFY_CHANGE_SIZE);
    if(event!=INVALID_HANDLE_VALUE){imageWrites_.setHandle(event);imageWrites_.setEnabled(true);}
}

void ProjectDiskWatch::saved(std::string fingerprint){
    ++generation_;
    fingerprint_=std::move(fingerprint);
    noteChange();
}

void ProjectDiskWatch::noteChange(){
    retries_=0;
    pending_=true;
    timer_.start(300);
}

void ProjectDiskWatch::check(){
    if(!pending_||reading_)return;
    if(!ready_()){timer_.start(750);return;}
    arm();
    pending_=false;
    reading_=true;
    readingGeneration_=generation_;
    reader_.setFuture(QtConcurrent::run([path=std::filesystem::path(path_.toStdWString()),known=fingerprint_]() -> std::optional<ObservedProject> {
        try{
            ProjectStore store(makeWicProjectCodec());
            return store.observe(path,known);
        }catch(const std::exception&){return std::nullopt;}
    }));
}

void ProjectDiskWatch::finish(){
    auto result=reader_.result();
    if(readingGeneration_==generation_){
        if(!result){
            // A cloud sync or directory swap can leave the package temporarily incomplete.
            if(++retries_<=5){pending_=true;timer_.start(250*(1<<retries_));}
        }else if(result->project){
            if(ready_()){
                fingerprint_=std::move(result->fingerprint);
                changed_(std::move(*result->project));
            }else pending_=true;
        }
    }
    reading_=false;
    if(pending_&&!timer_.isActive())timer_.start(300);
}

}
