#pragma once

#include <QString>
#include <QStringList>

class QMenu;
class QMenuBar;
class QWidget;

namespace compositor::ui {

void initializeInterfaceLanguage();
QString localized(const QString& source);
QStringList localizedList(const QStringList& source);
QMenu* localizedMenu(QMenuBar* parent, const QString& source);
QMenu* localizedMenu(QMenu* parent, const QString& source);
QString menuSourceTitle(const QMenu* menu);
void addLanguageMenu(QMenu* help, QWidget* owner);

}
