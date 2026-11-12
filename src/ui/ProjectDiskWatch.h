#pragma once

#include "persistence/ProjectStore.h"
#include <QFileSystemWatcher>
#include <QFutureWatcher>
#include <QTimer>
#include <QWinEventNotifier>
#include <functional>

namespace compositor::ui {

class ProjectDiskWatch final : public QObject {
public:
    ProjectDiskWatch(QString path,std::string fingerprint,std::function<bool()> ready,
                     std::function<void(OpenProject)> changed,QObject* parent);
    ~ProjectDiskWatch() override;
    const QString& path() const { return path_; }
    void saved(std::string fingerprint);

private:
    void arm();
    void noteChange();
    void check();
    void finish();
    void closeImageWatch();

    QString path_;
    std::string fingerprint_;
    std::function<bool()> ready_;
    std::function<void(OpenProject)> changed_;
    QFileSystemWatcher filesystem_;
    QWinEventNotifier imageWrites_;
    QTimer timer_;
    QFutureWatcher<std::optional<ObservedProject>> reader_;
    int generation_{}, readingGeneration_{}, retries_{};
    bool pending_{}, reading_{};
};

}
